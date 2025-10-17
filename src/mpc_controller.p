import casadi as ca
import numpy as np
import math # Use math for numpy constants like pi

class MPCController:
    """
    Model Predictive Controller (MPC) for Glider Thermal Soaring.
    Uses CasADi for efficient trajectory optimization.
    
    This version includes stability fixes and cost function tuning to encourage 
    thermal centering and altitude maximization.
    """
    def __init__(self, N=20, DT=1.0, glider_params=None):
        
        # Configuration
        self.N = N       # Prediction horizon (steps)
        self.DT = DT     # Time step (seconds)
        self.MAX_ITER = 200 # Increased iterations for better convergence
        self.EPSILON_AIRSPEED = 1e-4 # Stability constant to prevent division by zero
        
        # Glider Parameters loaded from the dictionary
        self.m = glider_params.get('mass', 700.0)
        self.g = 9.81
        self.rho = 1.225
        self.S = glider_params.get('S', 14.0)
        self.CD0 = glider_params.get('CD0', 0.015)
        self.K = glider_params.get('K', 0.04)
        self.CL = 0.8 # Placeholder CL matching GliderDynamics

        # Control Constraints (Radians)
        self.MAX_BANK = np.deg2rad(45)   
        self.MIN_BANK = np.deg2rad(-45)
        self.MAX_PITCH = np.deg2rad(10) 
        self.MIN_PITCH = np.deg2rad(-10)

        self.MIN_Z = 100.0 # Safety altitude constraint

        self._setup_casadi_solver()
        
        # CRITICAL FIX: Initial guess/solution for warm-starting the solver
        # Initialize with zeros (solver will optimize this from the first step)
        self.u_opt = np.zeros((2, N))
        self.x_opt = np.zeros((7, N + 1))


    def _setup_casadi_solver(self):
        # State: x=[x, y, z, vx, vy, vz, m], Control: u=[phi, gamma], Parameters: p=[cx, cy, r]
        self.x = ca.MX.sym('x', 7) 
        self.u = ca.MX.sym('u', 2) 
        self.p = ca.MX.sym('p', 3) 

        # Unpack symbolic variables
        _, _, z, vx, vy, vz, m = ca.vertsplit(self.x)
        phi, gamma = ca.vertsplit(self.u)
        thermal_cx, thermal_cy, thermal_radius = ca.vertsplit(self.p)

        # --- 1. Dynamic Model (EOM) in CasADi ---

        # Thermal Wind (W_atm = [0, 0, W_z])
        dist_sq = (self.x[0] - thermal_cx)**2 + (self.x[1] - thermal_cy)**2
        dist = ca.sqrt(dist_sq)

        W_z_max = 5.0
        # Thermal model: W_z is 0 if outside radius, or cosine profile if inside
        W_z = ca.if_else(dist < thermal_radius, 
                              W_z_max * (ca.cos(math.pi * dist / thermal_radius) + 1.0) / 2.0, 
                              0.0)
        W_atm_z = W_z
        
        # Airspeed Calculation (V_air = V_ground - W_atm)
        V_ground_vec = ca.vertcat(vx, vy, vz)
        W_atm_vec = ca.vertcat(0.0, 0.0, W_atm_z)
        
        V_air_vec = V_ground_vec - W_atm_vec
        V_air_sq = ca.dot(V_air_vec, V_air_vec)
        
        # CRITICAL FIX (NaN Stability): Ensure V_air_mag is non-zero
        V_air_mag = ca.sqrt(ca.fmax(V_air_sq, self.EPSILON_AIRSPEED))
        
        # Aerodynamic Forces
        CL = self.CL 
        CD = self.CD0 + self.K * CL**2
        
        L = 0.5 * self.rho * self.S * CL * V_air_mag**2
        D = 0.5 * self.rho * self.S * CD * V_air_mag**2
        
        # Force Vectors
        e_v = V_air_vec / V_air_mag # Unit vector of V_air
        D_vec = -D * e_v
        
        # Simplified Lift projection (matches glider_dynamics.py)
        L_x = L * (ca.sin(phi) * ca.sin(gamma)) 
        L_y = L * (-ca.cos(phi) * ca.sin(gamma)) 
        L_z = L * (ca.cos(gamma)) 
        L_vec = ca.vertcat(L_x, L_y, L_z)
        
        # Total Force and Acceleration
        G_vec = ca.vertcat(0.0, 0.0, -m * self.g)
        F_total = L_vec + D_vec + G_vec
        a_vec = F_total / m
        
        # State Derivatives (dx/dt)
        x_dot = ca.vertcat(V_ground_vec, a_vec, 0.0) 
        
        # Euler integration
        x_next = self.x + self.DT * x_dot
        
        # CasADi function for dynamics
        self.casadi_dynamics = ca.Function('F', [self.x, self.u, self.p], [x_next], 
                                             ['x_k', 'u_k', 'p'], ['x_k+1'])
        
        # --- 2. Cost Function Setup (Objective) ---
        J = 0 
        # --- TUNED WEIGHTS ---
        Q_pos = 1e2  # Drive glider toward thermal center
        Q_alt = 1e3  # Maximize altitude (minimize -Z)
        R_cont = 1e-3 # Minimize control effort
        Q_vz_term = 1e3 # Terminal penalty on vertical velocity (prevent diving)

        opt_vars = []
        
        # Setup bounds structures (CRITICAL FIX: Array shapes for broadcasting)
        self.lbx = np.zeros((7 * (self.N + 1) + 2 * self.N, 1))
        self.ubx = np.zeros((7 * (self.N + 1) + 2 * self.N, 1))
        selfbg = np.zeros((7 * self.N, 1)) # Equality constraints (Dynamics = 0)
        self.ubg = np.zeros((7 * self.N, 1))

        # Initial state variable X0
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
            
            # Dynamics constraint: Xk_next_sym - Xk_next = 0
            G += [Xk_next_sym - Xk_next] 

            # Running Cost L(x, u)
            pos_cost = Q_pos * ((Xk_next_sym[0] - thermal_cx)**2 + (Xk_next_sym[1] - thermal_cy)**2)
            alt_cost = -Q_alt * Xk_next_sym[2] # Maximize Z
            cont_cost = R_cont * (Uk[0]**2 + Uk[1]**2)
            
            J += pos_cost + alt_cost + cont_cost

            Xk = Xk_next_sym
            
            # --- Set Bounds for Control and State Variables ---
            
            # Control Bounds U_k
            u_idx_start = 7 * (self.N + 1) + 2 * k
            u_idx_end = 7 * (self.N + 1) + 2 * (k + 1)
            
            # FIX: Reshape bounds to (2, 1) column vector
            control_min = np.array([self.MIN_BANK, self.MIN_PITCH]).reshape(2, 1)
            control_max = np.array([self.MAX_BANK, self.MAX_PITCH]).reshape(2, 1)

            self.lbx[u_idx_start: u_idx_end] = control_min
            self.ubx[u_idx_start: u_idx_end] = control_max
            
            # State Bounds X_k+1 (z >= MIN_Z)
            state_lb = [-1e6, -1e6, self.MIN_Z, -1e6, -1e6, -1e6, self.m]
            state_ub = [1e6, 1e6, 1e6, 1e6, 1e6, 1e6, self.m]
            
            # FIX: Reshape state bounds to (7, 1) column vector
            state_lb_col = np.array(state_lb).reshape(7, 1)
            state_ub_col = np.array(state_ub).reshape(7, 1)

            self.lbx[7 * (k + 1): 7 * (k + 2)] = state_lb_col
            self.ubx[7 * (k + 1): 7 * (k + 2)] = state_ub_col

        # Final terminal cost (Encourage centering and stable vertical velocity)
        J += Q_pos * ((Xk[0] - thermal_cx)**2 + (Xk[1] - thermal_cy)**2)
        J += Q_vz_term * (Xk[5]**2) # Minimize final vertical speed

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
                'print_level': 0, # Suppress solver output
                'acceptable_tol': 1e-4,
                'acceptable_obj_change_tol': 1e-4,
                'linear_solver': 'ma27'
            },
            'print_time': False,
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    def compute_control(self, current_state, thermal_params):
        """
        Solves the NLP problem for the optimal control sequence using warm-starting.
        """
        # Set bounds for the initial state X0 (equal to current state)
        self.lbx[:7] = current_state.reshape(-1, 1) 
        self.ubx[:7] = current_state.reshape(-1, 1)

        # Warm-start: Shift the previous optimal solution by one step
        u_init = np.hstack([self.u_opt[:, 1:], self.u_opt[:, -1:]]) 
        x_init = np.hstack([self.x_opt[:, 1:], self.x_opt[:, -1:]])
        
        # Construct the full guess vector
        x0_guess = np.concatenate([x_init.flatten(), u_init.flatten()]).reshape(-1, 1)
        
        # Fallback to zero guess if dimensions are wrong (e.g., first run)
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
        
        # Save the optimized trajectory for the next warm-start
        self.x_opt = opt_sol[:7 * (self.N + 1)].reshape((7, self.N + 1))
        self.u_opt = opt_sol[7 * (self.N + 1):].reshape((2, self.N))
        
        # Return the first control action (phi, gamma)
        return self.u_opt[:, 0]
