import casadi as ca
import numpy as np

class MPCController:
    """
    Model Predictive Controller (MPC) for Glider Thermal Soaring.
    Uses CasADi for efficient trajectory optimization.
    """
    def __init__(self, N=20, DT=1.0, glider_params=None):
        
        # Configuration
        self.N = N       # Prediction horizon (steps)
        self.DT = DT     # Time step (seconds)
        self.T = N * DT  # Total prediction time
        self.MAX_ITER = 200 
        self.EPSILON_AIRSPEED = 1e-4
        
        # Glider Parameters (must be consistent with GliderDynamics)
        self.m = glider_params.get('mass', 700.0)
        self.g = 9.81
        self.rho = 1.225
        self.S = glider_params.get('S', 14.0)
        self.CD0 = glider_params.get('CD0', 0.015)
        self.K = glider_params.get('K', 0.04)

        # Control Constraints
        self.MAX_BANK = np.deg2rad(45)   
        self.MIN_BANK = np.deg2rad(-45)
        self.MAX_PITCH = np.deg2rad(10)
        self.MIN_PITCH = np.deg2rad(-10)

        # State Constraints (Soft/Penalty applied)
        self.MIN_Z = 100.0

        self._setup_casadi_solver()
        
        # Initial guess/solution (to warm-start the solver)
        self.u_opt = np.zeros((2, N))
        self.u_opt[0, :] = np.deg2rad(10.0)
        
        self.x_opt = np.zeros((7, N + 1))


    def _setup_casadi_solver(self):
        # State: x = [x, y, z, vx, vy, vz, m] (7 states)
        self.x = ca.MX.sym('x', 7) 
        # Control: u = [phi, gamma] (2 controls: bank, pitch/AoA proxy)
        self.u = ca.MX.sym('u', 2) 
        # Parameters: p = [thermal_cx, thermal_cy, thermal_radius] (3 parameters)
        self.p = ca.MX.sym('p', 3) 

        # Unpack state, control, and parameters
        _, _, z, vx, vy, vz, m = ca.vertsplit(self.x)
        phi, gamma = ca.vertsplit(self.u)
        thermal_cx, thermal_cy, thermal_radius = ca.vertsplit(self.p)

        # --- 1. Dynamic Model (EOM) in CasADi ---

        # Atmospheric Wind Model (Simplified for Optimization)
        # W_atm = [0, 0, W_z_thermal]
        dist_sq = (self.x[0] - thermal_cx)**2 + (self.x[1] - thermal_cy)**2
        dist = ca.sqrt(dist_sq)

        W_z_max = 5.0 # Max uplift 
        W_z = ca.if_else(dist < thermal_radius, 
                              W_z_max * (ca.cos(np.pi * dist / thermal_radius) + 1.0) / 2.0, 
                              0.0)
        W_atm_z = W_z
        
        # Airspeed and Aerodynamics
        V_ground_vec = ca.vertcat(vx, vy, vz)
        W_atm_vec = ca.vertcat(0.0, 0.0, W_atm_z)
        
        V_air_vec = V_ground_vec - W_atm_vec
        V_air_sq = ca.dot(V_air_vec, V_air_vec)
        
        # CRITICAL STABILITY FIX
        V_air_mag = ca.sqrt(ca.fmax(V_air_sq, self.EPSILON_AIRSPEED))
        
        # Aerodynamic Forces (Lift and Drag, simplified model)
        CL = 0.8 # Placeholder CL
        CD = self.CD0 + self.K * CL**2
        
        L = 0.5 * self.rho * self.S * CL * V_air_mag**2
        D = 0.5 * self.rho * self.S * CD * V_air_mag**2
        
        # Force Vectors
        e_v = V_air_vec / V_air_mag # Unit vector of V_air
        D_vec = -D * e_v
        
        # Simplified Lift projection 
        L_x = L * (ca.sin(phi) * ca.sin(gamma)) 
        L_y = L * (-ca.cos(phi) * ca.sin(gamma)) 
        L_z = L * (ca.cos(gamma)) 
        L_vec = ca.vertcat(L_x, L_y, L_z)
        
        # Gravity
        G_vec = ca.vertcat(0.0, 0.0, -m * self.g)
        
        # Total Force and Acceleration
        F_total = L_vec + D_vec + G_vec
        a_vec = F_total / m
        
        # State Derivatives (dx/dt)
        x_dot = ca.vertcat(V_ground_vec, a_vec, 0.0) 
        
        # Euler integration: x_next = x_k + dt * x_dot
        x_next = self.x + self.DT * x_dot
        
        # CasADi function for dynamics
        self.casadi_dynamics = ca.Function('F', [self.x, self.u, self.p], [x_next], 
                                             ['x_k', 'u_k', 'p'], ['x_k+1'])
        
        # --- 2. Cost Function Setup (Objective) ---
        J = 0 
        # --- TUNED WEIGHTS: Balance Altitude and Position ---
        Q_pos = 1e2  # MODERATE: Drive toward thermal center
        Q_alt = 1e3  # HIGH: Maximize altitude (the primary objective)
        R_cont = 1e-3 # LOW: Allow small control movements
        Q_vz_term = 1e3 # Terminal penalty on vertical velocity
        # ---------------------------------------------

        opt_vars = []
        
        self.lbx = np.zeros((7 * (self.N + 1) + 2 * self.N, 1))
        self.ubx = np.zeros((7 * (self.N + 1) + 2 * self.N, 1))
        self.lbg = np.zeros((7 * self.N, 1)) # Equality constraints (dynamics)
        self.ubg = np.zeros((7 * self.N, 1))

        # Initial state 
        X0 = ca.MX.sym('X0', 7)
        opt_vars += [X0]
        
        Xk = X0
        G = [] 

        # Loop over the prediction horizon
        for k in range(self.N):
            Uk = ca.MX.sym('U_' + str(k), 2)
            opt_vars += [Uk]

            Xk_next = self.casadi_dynamics(Xk, Uk, self.p)
            
            Xk_next_sym = ca.MX.sym('X_' + str(k + 1), 7)
            opt_vars += [Xk_next_sym]
            
            # Dynamics constraint: G_k = Xk_next_sym - Xk_next = 0
            G += [Xk_next_sym - Xk_next] 

            # Running Cost L(x, u)
            pos_cost = Q_pos * ((Xk_next_sym[0] - thermal_cx)**2 + (Xk_next_sym[1] - thermal_cy)**2)
            alt_cost = -Q_alt * Xk_next_sym[2] 
            cont_cost = R_cont * (Uk[0]**2 + Uk[1]**2)
            
            J += pos_cost + alt_cost + cont_cost

            Xk = Xk_next_sym
            
            # --- Constraints and Bounds ---
            
            # Bounds for Control U_k
            u_idx_start = 7 * (self.N + 1) + 2 * k
            u_idx_end = 7 * (self.N + 1) + 2 * (k + 1)
            
            control_min = np.array([self.MIN_BANK, self.MIN_PITCH]).reshape(2, 1)
            control_max = np.array([self.MAX_BANK, self.MAX_PITCH]).reshape(2, 1)

            self.lbx[u_idx_start: u_idx_end] = control_min
            self.ubx[u_idx_start: u_idx_end] = control_max
            
            # Bounds for State X_k+1
            state_lb = [-1e6, -1e6, self.MIN_Z, -1e6, -1e6, -1e6, self.m]
            state_ub = [1e6, 1e6, 1e6, 1e6, 1e6, 1e6, self.m]
            
            state_lb_col = np.array(state_lb).reshape(7, 1)
            state_ub_col = np.array(state_ub).reshape(7, 1)

            self.lbx[7 * (k + 1): 7 * (k + 2)] = state_lb_col
            self.ubx[7 * (k + 1): 7 * (k + 2)] = state_ub_col

        # Final terminal cost
        J += Q_pos * ((Xk[0] - thermal_cx)**2 + (Xk[1] - thermal_cy)**2)
        J += Q_vz_term * (Xk[5]**2)

        self.opt_vars = ca.vertcat(*opt_vars)
        self.G = ca.vertcat(*G)
        
        # --- 3. Solver Setup ---

        nlp = {
            'f': J,             
            'x': self.opt_vars, 
            'g': self.G,        
            'p': self.p         
        }
        
        opts = {
            'ipopt': {
                'max_iter': self.MAX_ITER,
                'print_level': 3,  
                'acceptable_tol': 1e-4,
                'acceptable_obj_change_tol': 1e-4,
                'linear_solver': 'ma57'  # <--- FIXED: Using MA57 instead of MA27 (due to missing HSL library)
            },
            'print_time': False,
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    def compute_control(self, current_state, thermal_params):
        """
        Solves the NLP problem and returns the optimal control sequence.
        thermal_params = [thermal_cx, thermal_cy, thermal_radius]
        """
        # Set bounds for the initial state X0 (equal to current state)
        self.lbx[:7] = current_state.reshape(-1, 1) if current_state.ndim == 1 else current_state
        self.ubx[:7] = current_state.reshape(-1, 1) if current_state.ndim == 1 else current_state

        # Initial guess (Warm-start)
        if self.u_opt.shape == (2, self.N):
            u_init = np.hstack([self.u_opt[:, 1:], self.u_opt[:, -1:]]) 
        else:
            u_init = np.zeros((2, self.N))
            u_init[0, :] = np.deg2rad(10.0) 
        
        if self.x_opt.shape == (7, self.N + 1):
            x_init = np.hstack([self.x_opt[:, 1:], self.x_opt[:, -1:]])
        else:
            x_init = np.zeros((7, self.N + 1))
        
        x0_guess = np.concatenate([x_init.flatten(), u_init.flatten()]).reshape(-1, 1)
        
        if x0_guess.shape[0] != self.opt_vars.shape[0]:
               x0_guess = np.zeros((self.opt_vars.shape[0], 1))
        
        # Run the solver
        sol = self.solver(
            x0=x0_guess,     
            lbx=self.lbx,    
            ubx=self.ubx,    
            lbg=self.lbg,    
            ubg=self.ubg,    
            p=thermal_params 
        )

        opt_sol = sol['x'].full().flatten()
        
        self.x_opt = opt_sol[:7 * (self.N + 1)].reshape((7, self.N + 1))
        self.u_opt = opt_sol[7 * (self.N + 1):].reshape((2, self.N))
        
        # Return the first control action (phi, gamma)
        return self.u_opt[:, 0]

