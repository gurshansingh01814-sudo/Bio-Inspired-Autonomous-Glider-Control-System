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
        self.N = mpc_params.get('horizon_steps', 40)      # Prediction horizon
        self.DT = mpc_params.get('control_rate_s', 2.0) # Control Time step (seconds)
        
        # Unpack Glider/Control Limits
        glider_limits = self.config.get('GLIDER', {})
        self.MAX_BANK_RAD = 0.523599
        self.CL_MIN = 0.4 
        self.CL_MAX = 1.2 
        self.V_MIN = 6.5 

        # Unpack Glider parameters (for symbolic dynamics)
        glider_params = self.config.get('GLIDER', {})
        env_params = self.config.get('ATMOSPHERE', {})
        self.m = ca.MX(glider_params.get('mass', 20.0))
        self.S = ca.MX(glider_params.get('S', 14.0))
        self.CD0 = ca.MX(glider_params.get('CD0', 0.015))
        self.K = ca.MX(glider_params.get('K', 0.04))
        self.g = ca.MX(env_params.get('gravity', 9.81))
        self.rho = ca.MX(env_params.get('rho_air', 1.225))
        
        self.EPSILON_AIRSPEED = 1e-4 
        self.EPSILON_LIFT = 1e-3     

        # Problem Setup (State and Control Dimensions)
        self.NX = 6 # [x, y, z, vx, vy, vz]
        self.NU = 2 # [CL, phi] 
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
        CL, phi = U[0], U[1] 
        
        V_ground = ca.vertcat(vx, vy, vz)
        W_in = ca.vertcat(ca.MX(0.0), ca.MX(0.0), W_atm_z)
        
        # Air velocity calculation
        V_air_vec = V_ground - W_in
        V_air_mag = ca.norm_2(V_air_vec)
        V_reg = ca.sqrt(V_air_mag**2 + self.EPSILON_AIRSPEED**2) 
        e_v = V_air_vec / V_reg 
        
        # 1. Drag Force (Fd)
        CD = self.CD0 + self.K * CL**2 
        D_mag = 0.5 * self.rho * self.S * CD * V_reg**2
        F_drag = -D_mag * e_v 
        
        # 2. Lift Force (Fl)
        L_mag = 0.5 * self.rho * self.S * CL * V_reg**2
        
        e_z = ca.vertcat(0.0, 0.0, 1.0)
        
        # Projection and perpendicular vectors
        dot_product = ca.dot(e_z, e_v)
        e_L_raw = e_z - dot_product * e_v 
        
        L_raw_mag_sq_reg = ca.sumsqr(e_L_raw) + self.EPSILON_LIFT**2
        L_raw_mag_reg = ca.sqrt(L_raw_mag_sq_reg)

        L_vert_unit = e_L_raw / L_raw_mag_reg
        
        L_side_unit = ca.cross(e_v, L_vert_unit) 

        # Lift Vector
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
        S_stall = opti.variable(1, self.N + 1) 
        
        V_air_sq_ground = X[3, :]**2 + X[4, :]**2 + X[5, :]**2
        
        V_TARGET = 10.0 # m/s  

        # Parameters
        P_init = opti.parameter(self.NX, 1)
        P_target = opti.parameter(2, 1) 
        P_Wz = opti.parameter(1, self.N) 
        
        # Objective Function
        J = 0 
        
        # Define Tuning Weights
        W_CLIMB = 200.0     
        W_SMOOTH = 50.0    
        W_DIST = 15.0      
        W_SLACK = 5000.0 
        W_AIRSPEED = 500.0  
        self.W_STALL = 20000.0 # Penalty for V_MIN violation

        # Cost Loop and Step-wise Constraints
        for k in range(self.N):
            
            # 1. Bio-Inspired Climbing Objective: Maximize vertical speed (vz)
            J += -W_CLIMB * X[5, k+1] 
            
            # 2. Proximity/Centering Cost: Minimize squared distance to the thermal center
            dist_sq = (X[0, k] - P_target[0])**2 + (X[1, k] - P_target[1])**2
            J += W_DIST * dist_sq
            
            # 3. Control Effort / Smoothness
            J += W_SMOOTH * ca.sumsqr(U[:, k] - U[:, k-1])
            
            # --- Airspeed calculation and Cost update ---
            vx_k, vy_k, vz_k = X[3, k], X[4, k], X[5, k]
            W_atm_z_k = P_Wz[0, k]
            V_air_vec_k = ca.vertcat(vx_k, vy_k, vz_k) - ca.vertcat(0.0, 0.0, W_atm_z_k)
            V_air_sq_k = ca.sumsqr(V_air_vec_k) 
            V_air_mag_k = ca.sqrt(V_air_sq_k)

            # 4. Airspeed Regulation Cost
            J += W_AIRSPEED * (V_air_mag_k - V_TARGET)**2 
            
            # 5. Continuity Constraint (Dynamic Model)
            X_dot = self._glider_dynamics(X[:, k], U[:, k], P_Wz[0, k])
            opti.subject_to(X[:, k+1] == X[:, k] + self.DT * X_dot)

            # --- Soft Minimum Airspeed Constraint ---
            
            # 1. Enforce minimum air velocity squared using slack
            opti.subject_to(V_air_sq_k >= self.V_MIN**2 - S_stall[0, k])
            opti.subject_to(S_stall[0, k] >= 0.0)
            
            # 2. Add high penalty to the objective function for any violation
            J += self.W_STALL * S_stall[0, k] 

            # Enforce maximum air velocity squared
            V_MAX = 50.0
            opti.subject_to(V_air_sq_k <= V_MAX**2) 
            
            # --- Terminal/Final State Cost (N+1 step) ---
        J += W_DIST * ((X[0, self.N] - P_target[0])**2 + (X[1, self.N] - P_target[1])**2)
        J += W_SLACK * ca.sumsqr(S_alt)
        J += self.W_STALL * S_stall[0, self.N] 

        # --- Constraints ---
        opti.subject_to(X[:, 0] == P_init)
        
        # Control Bounds (CL and Phi)
        opti.subject_to(opti.bounded(self.CL_MIN, U[0, :], self.CL_MAX)) 
        opti.subject_to(opti.bounded(-self.MAX_BANK_RAD, U[1, :], self.MAX_BANK_RAD)) 
        
        # State Bounds
        Z_MIN = 20.0 # meters
        opti.subject_to(X[2, :] >= Z_MIN - S_alt) 
        opti.subject_to(S_alt >= 0.0)
        
        # --- Final State V_MIN Constraint (Soft) ---
        vx_N, vy_N, vz_N = X[3, self.N], X[4, self.N], X[5, self.N]
        W_atm_z_N = ca.MX(0.0)
        
        V_ground_N = ca.vertcat(vx_N, vy_N, vz_N)
        W_in_N = ca.vertcat(0.0, 0.0, W_atm_z_N)
        V_air_vec_N = V_ground_N - W_in_N
        V_air_sq_N = ca.sumsqr(V_air_vec_N)

        # Enforce minimum air velocity squared on the final state (Soft Constraint)
        opti.subject_to(V_air_sq_N >= self.V_MIN**2 - S_stall[0, self.N])
        opti.subject_to(S_stall[0, self.N] >= 0.0)

        # Flight Path Angle Constraint
        MAX_GAMMA_RAD = math.radians(60.0) 
        MAX_GAMMA_SIN = ca.sin(MAX_GAMMA_RAD)
        
        V_air_sq_for_gamma = X[3, :]**2 + X[4, :]**2 + X[5, :]**2 
        
        # Upper bound: vz <= V_mag * sin(gamma_max)
        opti.subject_to(X[5, :] <= ca.sqrt(V_air_sq_for_gamma) * MAX_GAMMA_SIN)
        # Lower bound: vz >= -V_mag * sin(gamma_max)
        opti.subject_to(X[5, :] >= -ca.sqrt(V_air_sq_for_gamma) * MAX_GAMMA_SIN)
        
        # --- Initialization ---
        
        # 1. State Guess (X_guess)
        SAFE_X_GUESS_ARRAY = ca.DM([0.0, 0.0, 500.0, 7.0, 0.0, -1.0]) 
        X_safe_guess_DM = ca.repmat(SAFE_X_GUESS_ARRAY, 1, self.N + 1)
        
        opti.set_initial(X, X_safe_guess_DM) 
        
        # 2. Control Guess (U_guess)
        MODERATE_BANK_RAD = math.radians(15.0) 
        MODERATE_CL = 0.6
        U_safe = ca.DM([[MODERATE_CL], [MODERATE_BANK_RAD]]) 
        
        U_guess = ca.repmat(U_safe, 1, self.N) 
        
        opti.set_initial(U, U_guess) 
        opti.set_initial(S_alt, 0.0) 
        opti.set_initial(S_stall, 0.0) 

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
        P_all = np.concatenate([initial_state.flatten(), P_target.flatten(), P_Wz.flatten()])
        
        try:
            U_flat = self.solver(P_all)
            U_optimal = np.array(U_flat).reshape(self.NU, self.N)
            return U_optimal[:, 0] 
            
        except Exception as e:
            # Fallback to a feasible, minimum-drag glide command (CL_MIN = 0.4, Phi = 0.0)
            print(f"MPC Solver failed to converge: {e}. Falling back to safe glide.")
            return np.array([self.CL_MIN, 0.0])