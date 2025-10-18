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
        self.CL_MIN = 0.4 # Min lift coefficient (fast, low-drag glide)
        self.CL_MAX = 1.4 # Max lift coefficient (slow, high-lift thermal circle)

        # Unpack Glider parameters (for symbolic dynamics)
        glider_params = self.config.get('GLIDER', {})
        env_params = self.config.get('ATMOSPHERE', {})
        self.m = ca.MX(glider_params.get('mass', 20.0))
        self.S = ca.MX(glider_params.get('S', 14.0))
        self.CD0 = ca.MX(glider_params.get('CD0', 0.015))
        self.K = ca.MX(glider_params.get('K', 0.04))
        self.g = ca.MX(env_params.get('gravity', 9.81))
        self.rho = ca.MX(env_params.get('rho_air', 1.225))
        
        # CRITICAL NUMERICAL FIX: Increased epsilon for stability
        self.EPSILON_AIRSPEED = 1e-4 # Used for V_reg denominator
        self.EPSILON_LIFT = 1e-3     # Robust regularization for lift vector

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
        This function is numerically stabilized.
        """
        vx, vy, vz = X[3], X[4], X[5]
        CL, phi = U[0], U[1] # U[0] is CL, U[1] is phi
        
        V_ground = ca.vertcat(vx, vy, vz)
        W_in = ca.vertcat(ca.MX(0.0), ca.MX(0.0), W_atm_z)
        
        # Air velocity calculation
        V_air_vec = V_ground - W_in
        V_air_mag = ca.norm_2(V_air_vec)
        # Use robust V_reg for denominator stability
        V_reg = ca.sqrt(V_air_mag**2 + self.EPSILON_AIRSPEED**2) 
        e_v = V_air_vec / V_reg # Unit vector in direction of air velocity
        
        # 1. Drag Force (Fd)
        CD = self.CD0 + self.K * CL**2 
        D_mag = 0.5 * self.rho * self.S * CD * V_reg**2
        F_drag = -D_mag * e_v 
        
        # 2. Lift Force (Fl)
        L_mag = 0.5 * self.rho * self.S * CL * V_reg**2
        
        e_z = ca.vertcat(0.0, 0.0, 1.0)
        
        # Projection and perpendicular vectors (e_L_raw is perpendicular to e_v in the vertical plane)
        dot_product = ca.dot(e_z, e_v)
        e_L_raw = e_z - dot_product * e_v 
        
        # CRITICAL FIX: Numerically safer unit vector calculation for Lift
        L_raw_mag_sq_reg = ca.sumsqr(e_L_raw) + self.EPSILON_LIFT**2
        L_raw_mag_reg = ca.sqrt(L_raw_mag_sq_reg)

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
        S_alt = opti.variable(1, self.N + 1) # Slack for altitude constraint
        
        # Parameters
        P_init = opti.parameter(self.NX, 1)
        P_target = opti.parameter(2, 1) 
        P_Wz = opti.parameter(1, self.N) # Thermal lift (Wz) over the horizon
        
        # Objective Function
        J = 0 
        
        # Define Tuning Weights
        W_CLIMB = 5.0         
        W_SMOOTH = 0.001       
        W_DIST = 1.0          
        W_SLACK = 1000.0      

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
        
        # Velocity Bounds and Stability Constraint
        V_MIN = 10.0 
        V_MAX = 50.0 
        V_air_sq = X[3, :]**2 + X[4, :]**2 + X[5, :]**2
        V_mag = ca.sqrt(V_air_sq) # Airspeed magnitude
        
        # 1. Minimum and Maximum Airspeed constraint
        opti.subject_to(V_air_sq >= V_MIN**2) # Enforce minimum velocity squared
        opti.subject_to(V_mag <= V_MAX)
        
        # 2. CRITICAL FIX: Flight Path Angle Constraint for numerical stability (NO DIVISION)
        # This constraint is rewritten from: -sin(gamma_max) <= vz / V_mag <= sin(gamma_max)
        # To the numerically safe form: |vz| <= V_mag * sin(gamma_max)
        MAX_GAMMA_RAD = math.radians(60.0) 
        MAX_GAMMA_SIN = ca.sin(MAX_GAMMA_RAD)
        
        # Upper bound: vz <= V_mag * sin(gamma_max)
        opti.subject_to(X[5, :] <= V_mag * MAX_GAMMA_SIN)
        # Lower bound: vz >= -V_mag * sin(gamma_max)
        opti.subject_to(X[5, :] >= -V_mag * MAX_GAMMA_SIN)
        # This prevents NaN generation from division by small numbers near V_mag, 
        # as V_mag is guaranteed to be >= V_MIN.
        
        # -----------------------------------------------
        # --- CRITICAL FIX FOR MX/DM INITIALIZATION ---
        # -----------------------------------------------
        
        # 1. State Guess (X_guess): Use a feasible NUMERICAL placeholder (DM)
        # We use a numerical array corresponding to a legal, non-zero speed glide
        # [x, y, z, vx, vy, vz]
        SAFE_X_GUESS_ARRAY = ca.DM([0.0, 0.0, 500.0, 15.0, 0.0, -1.0])
        X_safe_guess_DM = ca.repmat(SAFE_X_GUESS_ARRAY, 1, self.N + 1)
        
        # set_initial must use a numerical DM for the guess value
        opti.set_initial(X, X_safe_guess_DM) 
        
        # 2. Control Guess (U_guess): Fixed Feasible Values (DM)
        MODERATE_BANK_RAD = math.radians(15.0) 
        MODERATE_CL = 0.5 
        U_safe = ca.DM([[MODERATE_CL], [MODERATE_BANK_RAD]]) 
        
        U_guess = ca.repmat(U_safe, 1, self.N) 
        
        opti.set_initial(U, U_guess) 
        opti.set_initial(S_alt, 0.0) 
        # -------------------------------------------

        opti.minimize(J)
        
        opts = {
            'ipopt': {'max_iter': 100, 'print_level': 0, 'acceptable_tol': 1e-6, 'acceptable_obj_change_tol': 1e-4, 'tol': 1e-3},
            'print_time': 0,
        }
        opti.solver('ipopt', opts)

        # Function definition remains the same
        P = ca.vertcat(P_init, P_target, ca.reshape(P_Wz, self.N, 1))
        U_out = ca.vertcat(U)
        
        return opti.to_function('solver', [P], [U_out])

    def solve_mpc(self, initial_state, thermal_center, Wz_prediction):
        """Solves the MPC problem for the current time step."""
        
        P_target = np.array([thermal_center[0], thermal_center[1]])
        P_Wz = Wz_prediction.flatten()
        # Ensure all inputs are numpy arrays/lists before concatenation
        P_all = np.concatenate([initial_state.flatten(), P_target.flatten(), P_Wz.flatten()])
        
        try:
            U_flat = self.solver(P_all)
            U_optimal = np.array(U_flat).reshape(self.NU, self.N)
            return U_optimal[:, 0] 
        
        except Exception as e:
            # Fallback to a feasible, minimum-drag glide command (CL_MIN = 0.2, Phi = 0.0)
            print(f"MPC Solver failed to converge: {e}. Falling back to safe glide.")
            # Note: This fallback should use the CL_MIN defined in the class
            return np.array([self.CL_MIN, 0.0])