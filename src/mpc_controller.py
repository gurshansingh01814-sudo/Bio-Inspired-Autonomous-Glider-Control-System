import casadi as ca
import numpy as np
import yaml
import os
import sys

class MPCController:
    
    def __init__(self, full_config_path):
        self.config = self._load_config(full_config_path)
        
        # Glider and Environment Parameters (omitted for brevity)
        self.g = 9.81
        self.rho = 1.225 
        glider_params = self.config.get('GLIDER', {})
        self.m = glider_params.get('mass', 700.0)
        self.S = glider_params.get('S', 14.0)
        self.CD0 = glider_params.get('CD0', 0.015)
        self.K = glider_params.get('K', 0.04)
        self.CL = glider_params.get('CL', 0.8) 
        
        mpc_params = self.config.get('MPC', {})
        self.N = mpc_params.get('N', 8) 
        self.DT = mpc_params.get('DT', 2.0) # Synchronized DT from config
        default_Qx = [5.0, 5.0, 1.0, 0.1, 0.1, 0.1] 
        self.Q_x = np.diag(mpc_params.get('STATE_WEIGHTS', default_Qx))
        self.R_u = np.diag(mpc_params.get('CONTROL_WEIGHTS', [0.1, 0.1]))
        
        # Alt Min not strictly needed if only using soft constraint penalty
        self.ALT_MIN = 20.0 
        self.EPSILON_AIRSPEED = 1e-6 

        self.solver = self._setup_solver()
        
        # Warm start storage
        self.U_prev = np.zeros((2, self.N))
        self.X_prev = None 
        self.last_successful_u = np.array([0.0, 0.0])


    def _load_config(self, path):
        """Loads configuration from a YAML file."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            # We already handled the path error in main_simulation.py
            print(f"FATAL: Could not load configuration file {path}: {e}")
            sys.exit(1)

    def _setup_dynamics(self):
        """Defines the symbolic non-linear glider dynamics using CasADi."""
        
        # --- State (X) and Control (U) Variables ---
        x = ca.SX.sym('x'); y = ca.SX.sym('y'); z = ca.SX.sym('z')
        vx = ca.SX.sym('vx'); vy = ca.SX.sym('vy'); vz = ca.SX.sym('vz')
        X = ca.vertcat(x, y, z, vx, vy, vz)
        
        phi = ca.SX.sym('phi'); alpha = ca.SX.sym('alpha')
        U = ca.vertcat(phi, alpha)
        
        Wx = ca.SX.sym('Wx'); Wy = ca.SX.sym('Wy'); Wz = ca.SX.sym('Wz')
        W_atm_casadi = ca.vertcat(Wx, Wy, Wz)
        
        # --- Dynamics (RK4 integration definition) ---
        V_ground = ca.vertcat(vx, vy, vz)
        V_air_vec = V_ground - W_atm_casadi
        V_air_mag = ca.sqrt(ca.sumsqr(V_air_vec))
        V_reg = ca.sqrt(V_air_mag**2 + self.EPSILON_AIRSPEED)
        e_v = V_air_vec / V_reg
        CD = self.CD0 + self.K * self.CL**2 
        L_mag = 0.5 * self.rho * self.S * self.CL * V_reg**2
        D_mag = 0.5 * self.rho * self.S * CD * V_reg**2
        D_vec = -D_mag * e_v
        
        # This is the LIFT definition that MUST MATCH glider_dynamics.py
        L_x = ca.if_else(ca.fabs(V_reg) < 1e-3, 0.0, L_mag * ca.sin(phi) * e_v[1] / V_reg * V_reg)
        L_y = ca.if_else(ca.fabs(V_reg) < 1e-3, 0.0, L_mag * -ca.sin(phi) * e_v[0] / V_reg * V_reg)
        L_z_pitch = L_mag * ca.cos(phi) * ca.cos(alpha) 
        L_vec = ca.vertcat(L_x, L_y, L_z_pitch)
        
        G_vec = ca.vertcat(0.0, 0.0, -self.m * self.g)
        F_total = L_vec + D_vec + G_vec
        a_vec = F_total / self.m
        X_dot = ca.vertcat(V_ground, a_vec)
        f = ca.Function('f', [X, U, W_atm_casadi], [X_dot])
        
        M = 4 
        DT_RK4 = self.DT / M
        X_k = X
        # RK4 integration
        for _ in range(M):
            k1 = f(X_k, U, W_atm_casadi)
            k2 = f(X_k + DT_RK4/2 * k1, U, W_atm_casadi)
            k3 = f(X_k + DT_RK4/2 * k2, U, W_atm_casadi)
            k4 = f(X_k + DT_RK4 * k3, U, W_atm_casadi)
            X_k = X_k + DT_RK4/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        F_map = ca.Function('F_map', [X, U, W_atm_casadi], [X_k], 
                            ['x_in', 'u_in', 'p_in'], ['x_next'])

        return X.shape[0], U.shape[0], F_map

    def _setup_solver(self):
        """Sets up the CasADi optimization problem (NLP)."""
        
        NX, NU, F_map = self._setup_dynamics()
        
        # --- 1. NLP Setup: Decision Variables ---
        opti = ca.Opti()
        U = opti.variable(NU, self.N)       
        X = opti.variable(NX, self.N + 1)   
        S_alt = opti.variable(1, self.N + 1) # Altitude Slack (for Min Alt)
        S_vel = opti.variable(1, self.N + 1) # Airspeed Slack (for Min Velocity)
        
        opti.subject_to(S_alt >= 0)
        opti.subject_to(S_vel >= 0)
        
        P = opti.parameter(NX + 3) 
        X_current = P[:NX]
        Thermal_target = P[NX:] 

        # --- 2. Constraints ---
        opti.subject_to(X[:, 0] == X_current)
        
        # DYNAMICS CONSTRAINT MODIFIED: NO HARD CONSTRAINT
        # for k in range(self.N):
        #     opti.subject_to(X[:, k+1] == F_map(X[:, k], U[:, k], ca.vertcat(0, 0, 0)))
        
        DEG_TO_RAD = ca.pi / 180.0
        
        # Control Bounds:
        opti.subject_to(opti.bounded(-45.0 * DEG_TO_RAD, U[0, :], 45.0 * DEG_TO_RAD)) # Bank Angle
        opti.subject_to(opti.bounded(-5.0 * DEG_TO_RAD, U[1, :], 15.0 * DEG_TO_RAD))  # Effective Pitch

        # State Bounds (Soft Altitude and Soft Min Velocity):
        for k in range(self.N + 1):
             opti.subject_to(X[2, k] + S_alt[0, k] >= self.ALT_MIN) 
        
        V_sq = X[3, :]**2 + X[4, :]**2 + X[5, :]**2
        opti.subject_to(V_sq + S_vel[0, :] >= 4.5**2) 
        opti.subject_to(V_sq <= 30.0**2) 
        
        # --- 3. Objective Function ---
        J = 0 
        P_DYN_COST = 100000.0 # CRITICAL: Very high penalty for dynamics violation
        
        for k in range(self.N):
            # Control Cost
            J += ca.mtimes([U[:, k].T, self.R_u, U[:, k]]) 
            
            # Altitude Maximization
            J += -300 * X[2, k+1] 
            
            # Target Attraction (Weight 1.0)
            dist_sq = (X[0, k+1] - Thermal_target[0])**2 + (X[1, k+1] - Thermal_target[1])**2
            J += 1.0 * dist_sq 
            
            # CRITICAL FIX: Dynamics Penalty (Soft Equality)
            E_dyn = X[:, k+1] - F_map(X[:, k], U[:, k], ca.vertcat(0, 0, 0))
            J += P_DYN_COST * ca.sumsqr(E_dyn)

        # Slack Penalties
        J += 100 * ca.sumsqr(S_alt) 
        J += 500 * ca.sumsqr(S_vel) 

        opti.minimize(J)
        
        # --- 4. Solver Options (Allow more iterations for high-cost problem) ---
        opts = {
            'ipopt': {
                'max_iter': 5000, # Increased iterations
                'print_level': 0, 
                'acceptable_tol': 1e-4, # Relaxed tolerance slightly
                'acceptable_obj_change_tol': 1e-6,
                'max_cpu_time': 1.95, 
            },
            'print_time': False,
        }
        opti.solver('ipopt', opts)
        
        self.opti = opti
        self.X = X
        self.U = U
        self.P = P
        
        return opti.to_function('solver', [P], [U])

    def compute_control(self, current_state, thermal_center, thermal_radius):
        """
        Computes the next optimal control input based on the current state.
        """
        
        p_val = np.hstack((current_state, thermal_center, thermal_radius)).flatten()
        self.opti.set_value(self.P, p_val)
        
        # Warm Start initialization (Keeping the last successful one)
        if self.U_prev is not None:
             self.opti.set_initial(self.U, self.U_prev)
             
        if self.X_prev is not None:
            self.opti.set_initial(self.X, self.X_prev)
        else:
             # Cold Start
             initial_x_guess = np.tile(current_state, (self.N + 1, 1)).T
             self.opti.set_initial(self.X, initial_x_guess)
        
        # --- Solve NLP ---
        try:
            # Create a solver instance (necessary for the first solve)
            sol = self.opti.solve() 
            u_star = sol.value(self.U)
            x_star = sol.value(self.X)
            
            # Update warm start (Shift previous solution)
            self.U_prev = np.hstack((u_star[:, 1:], u_star[:, -1:])) 
            self.X_prev = np.hstack((x_star[:, 1:], x_star[:, -1:])) 
            
            self.last_successful_u = u_star[:, 0]

            return u_star[:, 0] 
            
        except Exception as e:
            # If solver fails (Infeasible or CPU Time Exceeded)
            print("\nWARNING: IPOPT failed to converge. Returning last known successful control input.")
            print(f"Error: {e}")
            
            # CRITICAL FIX: DO NOT reset warm start on failure!
            # The *stale* warm start is still better than a cold start guess.
            
            return self.last_successful_u