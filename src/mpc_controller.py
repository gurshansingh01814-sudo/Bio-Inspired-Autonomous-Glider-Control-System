import casadi as ca
import numpy as np
import yaml
import os
import sys

class MPCController:
    """
    Implements a Model Predictive Controller using CasADi/IPOPT 
    to guide the glider toward thermal lift sources while managing altitude.
    """
    
    def __init__(self, full_config_path):
        # 1. Configuration Loading
        self.config = self._load_config(full_config_path)
        
        # 2. Glider and Environment Parameters (from config)
        self.g = 9.81
        self.rho = 1.225 # Air density
        
        glider_params = self.config.get('GLIDER', {})
        self.m = glider_params.get('mass', 700.0)
        self.S = glider_params.get('S', 14.0)
        self.CD0 = glider_params.get('CD0', 0.015)
        self.K = glider_params.get('K', 0.04)
        self.CL = glider_params.get('CL', 0.8) # Constant Lift Coefficient
        
        mpc_params = self.config.get('MPC', {})
        # Horizon maintained at N=8
        self.N = 8 
        self.DT = mpc_params.get('PREDICT_DT', 1.0) # Should load 2.0 from config
        self.Q_x = np.diag(mpc_params.get('STATE_WEIGHTS', [10.0, 10.0, 1.0, 0.1, 0.1, 0.1]))
        self.R_u = np.diag(mpc_params.get('CONTROL_WEIGHTS', [0.1, 0.1]))
        self.ALT_MIN = mpc_params.get('MIN_ALTITUDE', 20.0)
        self.EPSILON_AIRSPEED = 1e-6 # CRITICAL: For numerical stability in CasADi

        # 3. Setup CasADi Problem (Dynamics, Solver)
        self.solver = self._setup_solver()
        
        # 4. Storage for initial guess (warm start)
        self.U_prev = np.zeros((2, self.N))
        self.X_prev = np.zeros((6, self.N + 1))
        # Store the last successful control for emergency fallback
        self.last_successful_u = np.array([0.0, 0.0])


    def _load_config(self, path):
        """Loads configuration from a YAML file."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"FATAL: Could not load configuration file {path}: {e}")
            sys.exit(1)

    def _setup_dynamics(self):
        """Defines the symbolic non-linear glider dynamics using CasADi."""
        
        # --- State (X) and Control (U) Variables ---
        # X: [x, y, z, vx, vy, vz] (Position and Velocity)
        x = ca.SX.sym('x'); y = ca.SX.sym('y'); z = ca.SX.sym('z')
        vx = ca.SX.sym('vx'); vy = ca.SX.sym('vy'); vz = ca.SX.sym('vz')
        X = ca.vertcat(x, y, z, vx, vy, vz)
        NX = X.shape[0]

        # U: [phi, alpha] (Bank Angle, Effective Pitch Angle)
        phi = ca.SX.sym('phi'); alpha = ca.SX.sym('alpha')
        U = ca.vertcat(phi, alpha)
        NU = U.shape[0]

        # Parameter (P): [Wx, Wy, Wz] (Atmospheric Wind Vector)
        Wx = ca.SX.sym('Wx'); Wy = ca.SX.sym('Wy'); Wz = ca.SX.sym('Wz')
        W_atm_casadi = ca.vertcat(Wx, Wy, Wz)
        
        # --- 1. Velocity Dynamics ---
        V_ground = ca.vertcat(vx, vy, vz)
        V_air_vec = V_ground - W_atm_casadi
        
        # Airspeed magnitude
        V_air_mag = ca.sqrt(ca.sumsqr(V_air_vec))
        
        # --- 2. CRITICAL: Regularization for Stability ---
        V_reg = ca.sqrt(V_air_mag**2 + self.EPSILON_AIRSPEED)
        
        # Unit vector of air velocity (e_v)
        e_v = V_air_vec / V_reg
        
        # --- 3. Aerodynamic Forces ---
        CD = self.CD0 + self.K * self.CL**2 
        L_mag = 0.5 * self.rho * self.S * self.CL * V_reg**2
        D_mag = 0.5 * self.rho * self.S * CD * V_reg**2
        
        # Drag Vector
        D_vec = -D_mag * e_v
        
        # Lift Vector (Simplified projection using controls)
        L_x = ca.if_else(ca.fabs(V_reg) < 1e-3, 0.0, L_mag * ca.sin(phi) * e_v[1] / V_reg)
        L_y = ca.if_else(ca.fabs(V_reg) < 1e-3, 0.0, L_mag * -ca.sin(phi) * e_v[0] / V_reg)
        L_z_pitch = L_mag * ca.cos(phi) * ca.cos(alpha) 
        L_vec = ca.vertcat(L_x, L_y, L_z_pitch)
        
        # --- 4. Total Force and Dynamics (F = ma) ---
        G_vec = ca.vertcat(0.0, 0.0, -self.m * self.g)
        
        # Total Force: Lift + Drag + Gravity
        F_total = L_vec + D_vec + G_vec
        
        # Acceleration
        a_vec = F_total / self.m

        # --- Final System Dynamics (X_dot = f(X, U, P)) ---
        X_dot = ca.vertcat(V_ground, a_vec)
        
        # Continuous-time function
        f = ca.Function('f', [X, U, W_atm_casadi], [X_dot])
        
        # Discrete-time map (Runge-Kutta 4)
        M = 4 # RK4 steps
        DT_RK4 = self.DT / M
        X_k = X
        for _ in range(M):
            k1 = f(X_k, U, W_atm_casadi)
            k2 = f(X_k + DT_RK4/2 * k1, U, W_atm_casadi)
            k3 = f(X_k + DT_RK4/2 * k2, U, W_atm_casadi)
            k4 = f(X_k + DT_RK4 * k3, U, W_atm_casadi)
            X_k = X_k + DT_RK4/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        F_map = ca.Function('F_map', [X, U, W_atm_casadi], [X_k], 
                            ['x_in', 'u_in', 'p_in'], ['x_next'])

        return NX, NU, F_map

    def _setup_solver(self):
        """Sets up the CasADi optimization problem (NLP)."""
        
        NX, NU, F_map = self._setup_dynamics()
        
        # --- 1. NLP Setup ---
        opti = ca.Opti()
        
        # Decision Variables
        U = opti.variable(NU, self.N)       
        X = opti.variable(NX, self.N + 1)   
        S = opti.variable(1, self.N + 1) # Slack variable (for soft altitude constraint)
        opti.subject_to(S >= 0)
        
        # Parameters
        P = opti.parameter(NX + 3) 
        X_current = P[:NX]
        Thermal_target = P[NX:] 

        # --- 2. Initial Condition Constraint ---
        opti.subject_to(X[:, 0] == X_current)
        
        # --- 3. Dynamic Constraints ---
        for k in range(self.N):
            opti.subject_to(X[:, k+1] == F_map(X[:, k], U[:, k], ca.vertcat(0, 0, 0)))
            
        # --- 4. Bounds and Constraints ---
        DEG_TO_RAD = ca.pi / 180.0
        
        # Control Bounds:
        opti.subject_to(opti.bounded(-45.0 * DEG_TO_RAD, U[0, :], 45.0 * DEG_TO_RAD)) # Bank Angle (phi)
        opti.subject_to(opti.bounded(-5.0 * DEG_TO_RAD, U[1, :], 10.0 * DEG_TO_RAD))  # Effective Pitch (alpha)

        # State Bounds:
        # Altitude Constraint (Softened): z + S >= ALT_MIN
        for k in range(self.N + 1):
             opti.subject_to(X[2, k] + S[0, k] >= self.ALT_MIN) 
        
        # Velocity Bounds 
        # CRITICAL FIX: Relax minimum airspeed for feasibility (from 5.0 to 4.5 m/s)
        opti.subject_to(X[3, :]**2 + X[4, :]**2 + X[5, :]**2 >= 4.5**2) 
        opti.subject_to(X[3, :]**2 + X[4, :]**2 + X[5, :]**2 <= 30.0**2)
        
        # --- 5. Objective Function ---
        J = 0 
        
        for k in range(self.N):
            # Minimize Control Effort 
            J += ca.mtimes([U[:, k].T, self.R_u, U[:, k]]) 

            # Primary Objective: Maximize altitude 
            J += -300 * X[2, k+1] 
            
            # Secondary Objective: Navigate towards the thermal 
            dist_sq = (X[0, k+1] - Thermal_target[0])**2 + (X[1, k+1] - Thermal_target[1])**2
            J += 0.01 * dist_sq 
        
        # Tertiary Objective: Penalize Slack Variable Use (Reduced penalty for feasibility)
        J += 100 * ca.sumsqr(S) 

        opti.minimize(J)
        
        # --- 6. Solver Options and Compilation ---
        opts = {
            'ipopt': {
                'max_iter': 3000, 
                'print_level': 0, 
                'acceptable_tol': 1e-4, 
                'acceptable_obj_change_tol': 1e-6,
                # CRITICAL FIX: Increase CPU time slightly for the 2.0s time step
                'max_cpu_time': 1.9, # Increased from 1.8 to 1.9
            },
            'print_time': False,
        }
        opti.solver('ipopt', opts)
        
        # Store the optimization problem structure
        self.opti = opti
        self.X = X
        self.U = U
        self.P = P
        
        return opti.to_function('solver', [P], [U])

    def compute_control(self, current_state, thermal_center, thermal_radius):
        """
        Computes the next optimal control input based on the current state.
        """
        
        # Set parameters (P)
        p_val = np.hstack((current_state, thermal_center, thermal_radius)).flatten()
        self.opti.set_value(self.P, p_val)
        
        # Warm start (re-use previous solution as initial guess)
        if self.U_prev is not None:
             self.opti.set_initial(self.U, self.U_prev)
             
        # Set an initial state trajectory guess
        if self.X_prev is not None:
            self.opti.set_initial(self.X, self.X_prev)
        else:
             # Fallback guess: Propagate current state forward
             self.opti.set_initial(self.X, np.tile(current_state, (self.N + 1, 1)).T)
        
        # --- Solve NLP ---
        try:
            sol = self.opti.solve()
            u_star = sol.value(self.U)
            x_star = sol.value(self.X)
            
            # Update warm start for next iteration (shift and repeat last)
            self.U_prev = np.hstack((u_star[:, 1:], u_star[:, -1:])) 
            self.X_prev = np.hstack((x_star[:, 1:], x_star[:, -1:])) 
            
            # Store successful control
            self.last_successful_u = u_star[:, 0]

            return u_star[:, 0] # Return the first control input
            
        except Exception as e:
            # If solver fails (e.g., local infeasibility, max iter, max CPU time)
            print("\nWARNING: IPOPT failed to converge. Returning last known successful control input.")
            print(f"Error: {e}")
            
            # Use the last successful control as a fallback
            return self.last_successful_u