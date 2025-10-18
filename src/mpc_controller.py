import casadi as ca
import numpy as np
import yaml
import sys
import math
import os 

class MPCController:

    def __init__(self, config_path):
        """Initializes the MPC problem, objective, constraints, and solver."""

        self.config = self._load_config(config_path)
        
        # Unpack Parameters
        mpc_params = self.config.get('MPC', {})
        self.N = mpc_params.get('horizon_steps', 40)    # Prediction horizon
        self.DT = mpc_params.get('control_rate_s', 2.0) # Control Time step (seconds)
        
        # Unpack Glider/Control Limits
        glider_limits = self.config.get('GLIDER', {})
        self.MAX_BANK_RAD = math.radians(glider_limits.get('max_bank_angle_deg', 45.0))
        self.CL_MIN = 0.2 # Min lift coefficient (fast, low-drag glide)
        self.CL_MAX = 1.2 # Max lift coefficient (slow, high-lift thermal circle)

        # Unpack Glider parameters (for symbolic dynamics)
        glider_params = self.config.get('GLIDER', {})
        env_params = self.config.get('ATMOSPHERE', {})
        self.m = ca.MX(glider_params.get('mass', 20.0))
        self.S = ca.MX(glider_params.get('S', 14.0))
        self.CD0 = ca.MX(glider_params.get('CD0', 0.015))
        self.K = ca.MX(glider_params.get('K', 0.04))
        self.g = ca.MX(env_params.get('gravity', 9.81))
        self.rho = ca.MX(env_params.get('rho_air', 1.225))
        self.EPSILON_AIRSPEED = 1e-6 # Used for V_reg denominator
        self.EPSILON_LIFT = 1e-4     # Robust regularization for lift vector

        # Problem Setup (State and Control Dimensions)
        self.NX = 6 # [x, y, z, vx, vy, vz]
        self.NU = 2 # [CL, phi] (Lift Coefficient, Bank Angle in radians) 
        self.solver = self._setup_solver()
        
    def _load_config(self, path):
        """Loads configuration from a YAML file for internal use."""
        if not os.path.exists(path):
            print(f"FATAL: Configuration file not found at {path}")
            sys.exit(1)
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"FATAL: Could not load configuration file {path}: {e}")
            sys.exit(1)
            
    def _glider_dynamics(self, X, U, W_atm_z):
        """
        Symbolic, continuous-time state-space model (X_dot = f(X, U, W_atm)).
        """
        vx, vy, vz = X[3], X[4], X[5]
        CL, phi = U[0], U[1] # U[0] is CL, U[1] is phi
        
        V_ground = ca.vertcat(vx, vy, vz)
        W_in = ca.vertcat(ca.MX(0.0), ca.MX(0.0), W_atm_z)
        
        # Air velocity calculation
        V_air_vec = V_ground - W_in
        V_air_mag = ca.norm_2(V_air_vec)
        V_reg = ca.sqrt(V_air_mag**2 + self.EPSILON_AIRSPEED)
        e_v = V_air_vec / V_reg # Unit vector in direction of air velocity
        
        # 1. Drag Force (Fd)
        CD = self.CD0 + self.K * CL**2              # Induced drag depends on control CL
        D_mag = 0.5 * self.rho * self.S * CD * V_reg**2
        F_drag = -D_mag * e_v                       
        
        # 2. Lift Force (Fl)
        L_mag = 0.5 * self.rho * self.S * CL * V_reg**2
        
        e_z = ca.vertcat(0.0, 0.0, 1.0)
        
        # Projection and perpendicular vectors
        dot_product = ca.dot(e_z, e_v)
        e_L_raw = e_z - dot_product * e_v 
        
        # CRITICAL FIX: Robust regularization for lift vector calculation
        L_raw_mag_reg = ca.norm_2(e_L_raw) + self.EPSILON_LIFT
        L_vert_unit = e_L_raw / L_raw_mag_reg
        
        L_side_unit = ca.cross(e_v, L_vert_unit) 

        # Lift Vector: Components are rotated by the bank angle (phi)
        F_lift = L_mag * (ca.cos(phi) * L_vert_unit + ca.sin(phi) * L_side_unit)

        # 3. Gravity Force (Fg)
        F_gravity = ca.vertcat(0.0, 0.0, -self.m * self.g)
        
        # Total Force and Acceleration (F = ma)
        F_total = F_lift + F_drag + F_gravity
        a_vec = F_total / self.m
        
        # X_dot = [V_ground, a_vec]
        X_dot = ca.vertcat(V_ground, a_vec)
        return X_dot

    def _setup_solver(self):
        """Sets up the CasADi optimization problem (OCP)."""
        
        opti = ca.Opti()

        # Decision variables
        X = opti.variable(self.NX, self.N + 1)
        U = opti.variable(self.NU, self.N)
        S_alt = opti.variable(1, self.N + 1)
        
        # Parameters
        P_init = opti.parameter(self.NX, 1)
        P_target = opti.parameter(2, 1) 
        P_Wz = opti.parameter(1, self.N) # Thermal lift (Wz) over the horizon
        
        # Objective Function
        J = 0 
        
        # Define Tuning Weights
        W_CLIMB = 1.0     
        W_SMOOTH = 0.01   
        W_DIST = 0.05     
        W_SLACK = 10000.0 

        # Cost Loop
        for k in range(self.N):
            
            # 1. Bio-Inspired Climbing Objective: Maximize vertical speed (vz)
            J += -W_CLIMB * X[5, k+1] 
            
            # 2. Proximity/Centering Cost: Minimize squared distance to the thermal center
            dist_sq = (X[0, k] - P_target[0])**2 + (X[1, k] - P_target[1])**2
            J += W_DIST * dist_sq
            
            # 3. Control Effort / Smoothness
            J += W_SMOOTH * ca.sumsqr(U[:, k])
            
            # 4. Continuity Constraint (Dynamic Model)
            X_dot = self._glider_dynamics(X[:, k], U[:, k], P_Wz[0, k])
            opti.subject_to(X[:, k+1] == X[:, k] + self.DT * X_dot)

        # --- Terminal/Final State Cost (N+1 step) ---
        J += W_DIST * ((X[0, self.N] - P_target[0])**2 + (X[1, self.N] - P_target[1])**2)
        J += W_SLACK * ca.sumsqr(S_alt)

        # --- Constraints ---
        opti.subject_to(X[:, 0] == P_init)
        
        # Control Bounds (CL and Phi)
        opti.subject_to(opti.bounded(self.CL_MIN, U[0, :], self.CL_MAX)) # U[0] is CL
        opti.subject_to(opti.bounded(-self.MAX_BANK_RAD, U[1, :], self.MAX_BANK_RAD)) # U[1] is Phi
        
        # State Bounds
        Z_MIN = 20.0 # meters
        opti.subject_to(X[2, :] >= Z_MIN - S_alt) # Altitude constraint with slack
        opti.subject_to(S_alt >= 0.0)
        
        # Velocity Bounds 
        V_MIN = 10.0 
        V_MAX = 50.0 
        V_air_sq = X[3, :]**2 + X[4, :]**2 + X[5, :]**2
        opti.subject_to(V_air_sq >= V_MIN**2)
        opti.subject_to(ca.sqrt(V_air_sq) <= V_MAX)
        
        # --- CRITICAL FIX: Symbolic Warm Start / Initial Guess ---
        # Solves the "MX object has no attribute 'full'" error and ensures feasibility.
        
        # 1. State Guess (X_guess): Repeat the initial state (P_init) across the entire horizon (N+1 steps)
        X_guess = ca.repmat(P_init, 1, self.N + 1)
        
        # 2. Control Guess (U_guess): Repeat the feasible control [CL_MIN, 0.0] across the horizon (N steps)
        U_safe = ca.vertcat(ca.DM(self.CL_MIN), ca.DM(0.0)) # Numerical safe control vector
        U_guess = ca.repmat(U_safe, 1, self.N)
        
        opti.set_initial(X, X_guess)
        opti.set_initial(U, U_guess)
        opti.set_initial(S_alt, 0.0) 
        # ------------------------------------------------

        opti.minimize(J)
        
        opts = {
            'ipopt': {'max_iter': 100, 'print_level': 0, 'acceptable_tol': 1e-6, 'acceptable_obj_change_tol': 1e-4, 'tol': 1e-3},
            'print_time': 0,
        }
        opti.solver('ipopt', opts)

        P = ca.vertcat(P_init, P_target, ca.reshape(P_Wz, self.N, 1))
        U_out = ca.vertcat(U)
        
        return opti.to_function('solver', [P], [U_out])

    def solve_mpc(self, initial_state, thermal_center, Wz_prediction):
        """Solves the MPC problem for the current time step."""
        
        P_target = np.array([thermal_center[0], thermal_center[1]])
        P_Wz = Wz_prediction.flatten()
        P_all = np.concatenate([initial_state, P_target, P_Wz])
        
        try:
            U_flat = self.solver(P_all)
            U_optimal = np.array(U_flat).reshape(self.NU, self.N)
            return U_optimal[:, 0] 
        
        except Exception as e:
            # Fallback to a feasible, minimum-drag glide command (CL_MIN = 0.2)
            return np.array([self.CL_MIN, 0.0])