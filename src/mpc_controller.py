import casadi as ca
import numpy as np
import yaml
import os

class MPCController:
    def __init__(self, config_path='data/glider_config.yaml'):
        
        # --- 1. Load Configuration ---
        try:
            # Assumes config file is in the data/ directory relative to the project root
            # Adjust path if your configuration file is elsewhere
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            full_config_path = os.path.join(project_root, config_path)
            with open(full_config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            self.config = {}
            # Provide sensible defaults if config fails
            self.config['MPC'] = {'N': 20, 'DT': 1.0, 'Q_diag': [1e3, 1e3, 5e3, 1e1, 1e1, 1e1], 'R_diag': [1e-1, 1e-1]}
            self.config['GLIDER'] = {'mass': 700.0, 'S': 14.0, 'CD0': 0.015, 'K': 0.04, 'CL': 0.8}
            self.config['CONSTRAINTS'] = {'MIN_AIRSPEED': 10.0, 'MAX_BANK_DEG': 45.0, 'MAX_PITCH_DEG': 10.0}
            
        # --- 2. MPC Parameters ---
        self.N = self.config['MPC']['N']
        self.DT = self.config['MPC']['DT']
        self.GLIDER_PARAMS = self.config['GLIDER']
        self.CONSTRAINTS = self.config['CONSTRAINTS']
        
        # --- 3. Stability Parameter (CRITICAL FIX) ---
        # Used to prevent division by zero in V_air calculation
        self.EPSILON_AIRSPEED = 1e-6 
        
        # --- 4. Initialize Optimization Problem ---
        self.solver = self._setup_optimization_problem()
        
        # --- 5. Warm-Start Storage ---
        # Used to store the solution from the previous step for faster convergence
        self.X_opt = None
        self.U_opt = None

    def _get_glider_dynamics(self, x, u, thermal_params):
        """
        Defines the continuous-time dynamics (dx/dt = f(x, u, p)) in CasADi symbolic form.
        State x = [x, y, z, vx, vy, vz] (6 states)
        Control u = [phi, alpha] (Bank angle, Angle of Attack - treated as effective pitch control)
        Parameter p = [cx, cy, radius] (Thermal parameters)
        """
        # --- Unpack State and Control ---
        x_pos, y_pos, z_pos, vx, vy, vz = ca.vertsplit(x)
        phi, alpha = ca.vertsplit(u)
        
        # --- Unpack Glider Parameters ---
        m = self.GLIDER_PARAMS['mass']
        g = 9.81
        rho = 1.225
        S = self.GLIDER_PARAMS['S']
        CD0 = self.GLIDER_PARAMS['CD0']
        K = self.GLIDER_PARAMS['K']
        CL = self.GLIDER_PARAMS['CL'] # Constant CL for dynamics model
        
        # --- Unpack Thermal Parameters ---
        cx, cy, radius = ca.vertsplit(thermal_params)

        # --- Atmospheric Wind (W_atm) ---
        
        # Distance to thermal center
        dist = ca.sqrt((x_pos - cx)**2 + (y_pos - cy)**2)
        
        # Uplift model (Wz) using a smooth cosine function
        # W_z_max is hardcoded here (3.0 m/s) but should ideally be in config/params
        W_z_max = 3.0 
        
        # Calculate Wz symbolically based on distance
        dist_ratio = dist / radius
        W_z = ca.if_else(dist < radius, W_z_max * (ca.cos(ca.pi * dist_ratio) + 1.0) / 2.0, 0.0)
        W_atm_vec = ca.vertcat(0.0, 0.0, W_z)
        
        # --- Airspeed and Unit Vector (V_air) ---
        V_ground_vec = ca.vertcat(vx, vy, vz)
        V_air_vec = V_ground_vec - W_atm_vec

        # V_air magnitude (CRITICAL FIX: Use regularization to prevent NaN)
        V_air_sq = ca.dot(V_air_vec, V_air_vec)
        V_reg = ca.sqrt(V_air_sq + self.EPSILON_AIRSPEED) # Stability Epsilon
        
        e_v = V_air_vec / V_reg # Unit vector of V_air (direction of relative wind)

        # --- Aerodynamic Forces ---
        CD = CD0 + K * CL**2
        
        L_mag = 0.5 * rho * S * CL * V_reg**2
        D_mag = 0.5 * rho * S * CD * V_reg**2
        
        # Drag Force opposes the relative wind direction
        D_vec = -D_mag * e_v
        
        # Lift Force simplified projection: phi controls direction, alpha/CL controls magnitude (simplified)
        # Note: This is a simplified 6DOF model aligned with the MPC formulation
        L_x = L_mag * ca.sin(phi) * ca.sin(alpha) 
        L_y = -L_mag * ca.cos(phi) * ca.sin(alpha) 
        L_z = L_mag * ca.cos(alpha) 
        L_vec = ca.vertcat(L_x, L_y, L_z)

        # --- Total Force & Acceleration ---
        G_vec = ca.vertcat(0.0, 0.0, -m * g)
        F_total = L_vec + D_vec + G_vec
        
        # State derivatives (velocities and accelerations)
        x_dot = ca.vertcat(V_ground_vec, F_total / m)
        
        return x_dot

    def _setup_optimization_problem(self):
        
        # --- 1. Setup Optimization Variables ---
        X = ca.SX.sym('X', 6, self.N + 1)  # State trajectory
        U = ca.SX.sym('U', 2, self.N)      # Control trajectory
        P = ca.SX.sym('P', 6 + 3)          # Parameters: [X0 (6), Thermal Params (3)]
        
        # --- 2. Cost Function and Constraints ---
        obj = 0  # Objective function
        g = []   # Constraints (list of expressions that must be zero)
        
        Q = ca.diag(self.config['MPC']['Q_diag'])  # State cost matrix
        R = ca.diag(self.config['MPC']['R_diag'])  # Control cost matrix
        
        # Initial state constraint (first constraint must match initial state P[0:6])
        g.append(X[:, 0] - P[0:6])
        
        # --- 3. Runge-Kutta 4 Integration (Time Discretization) ---
        for k in range(self.N):
            x_k = X[:, k]
            u_k = U[:, k]
            
            # --- Objective Contribution ---
            # Penalize deviation from thermal center (Target State is Thermal Center at max altitude)
            # The thermal center is fixed in the xy plane, but the target z is a constant high value
            X_target = ca.vertcat(P[6], P[7], 500.0, 0.0, 0.0, 0.0) # [cx, cy, target_z, 0, 0, 0]
            obj += ca.mtimes([(x_k - X_target).T, Q, (x_k - X_target)])
            obj += ca.mtimes([u_k.T, R, u_k]) # Minimize control effort

            # --- Dynamics (RK4) ---
            k1 = self._get_glider_dynamics(x_k, u_k, P[6:9])
            k2 = self._get_glider_dynamics(x_k + self.DT / 2 * k1, u_k, P[6:9])
            k3 = self._get_glider_dynamics(x_k + self.DT / 2 * k2, u_k, P[6:9])
            k4 = self._get_glider_dynamics(x_k + self.DT * k3, u_k, P[6:9])
            X_next = x_k + self.DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            
            # Dynamics constraint: X_{k+1} - X_next = 0
            g.append(X[:, k+1] - X_next)
            
        # --- Terminal Cost (Last State) ---
        X_target_N = ca.vertcat(P[6], P[7], 500.0, 0.0, 0.0, 0.0) 
        obj += ca.mtimes([(X[:, self.N] - X_target_N).T, Q, (X[:, self.N] - X_target_N)])


        # --- 4. Optimization Setup ---
        nlp = {'f': obj, 'x': ca.vertcat(ca.vec(X), ca.vec(U)), 'p': P, 'g': ca.vertcat(*g)}

        # --- 5. Solver Options (CRITICAL Tunning) ---
        # NOTE: print_level=0 for production, but 3 is good for initial debugging
        opts = {
            'ipopt': {
                'tol': 1e-4,              # RELAXED TOLERANCE (Stability Fix)
                'max_iter': 500,          # Increased iterations (Stability Fix)
                'linear_solver': 'mumps', # Robust linear solver
                'print_level': 3,         # Verbose output for debugging
                'acceptable_iter': 1,     # Accept a poor solution if no better can be found
            },
            'print_time': False,
        }

        # --- 6. Create Solver ---
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # --- 7. Bounds (lbx, ubx, lbu, ubu, lbg, ubg) ---
        self.lbx = np.zeros(6 * (self.N + 1) + 2 * self.N)
        self.ubx = np.zeros(6 * (self.N + 1) + 2 * self.N)
        self.lbg = np.zeros(6 * (self.N + 1))
        self.ubg = np.zeros(6 * (self.N + 1))

        # State Bounds
        # x, y, z, vx, vy, vz
        x_min, x_max = -5000.0, 5000.0
        y_min, y_max = -5000.0, 5000.0
        z_min, z_max = 5.0, 1000.0       # Must maintain altitude > 5m
        v_min = self.CONSTRAINTS['MIN_AIRSPEED'] # 10.0 m/s minimum airspeed
        v_max = 50.0

        for k in range(self.N + 1):
            idx = k * 6
            self.lbx[idx:idx+6] = [x_min, y_min, z_min, v_min, v_min, -v_max]
            self.ubx[idx:idx+6] = [x_max, y_max, z_max, v_max, v_max, v_max]
        
        # Control Bounds (rad)
        # phi: Bank angle | alpha: Effective pitch/AoA angle (simplified)
        max_phi = np.deg2rad(self.CONSTRAINTS['MAX_BANK_DEG'])   # 45 degrees
        max_alpha = np.deg2rad(self.CONSTRAINTS['MAX_PITCH_DEG']) # 10 degrees (Controls vertical force component)

        for k in range(self.N):
            u_idx = 6 * (self.N + 1) + k * 2
            self.lbx[u_idx:u_idx+2] = [-max_phi, -max_alpha]
            self.ubx[u_idx:u_idx+2] = [max_phi, max_alpha]
        
        return solver

    def compute_control(self, current_state, thermal_center, thermal_radius):
        """
        Computes the optimal control input (phi, alpha) for the current state.
        :param current_state: [x, y, z, vx, vy, vz] (numpy array)
        :param thermal_center: [cx, cy] (numpy array)
        :param thermal_radius: radius (float)
        :returns: Optimal control U[0] (phi, alpha)
        """
        
        # --- 1. Parameter Vector P (X0 and Thermal Params) ---
        thermal_params = np.array([thermal_center[0], thermal_center[1], thermal_radius])
        P_val = ca.vertcat(current_state, thermal_params)
        
        # --- 2. Initial Guess (Warm-Start Fix) ---
        if self.X_opt is None:
            # Cold Start: Initialize state trajectory with constant current state
            X_guess = np.tile(current_state, (self.N + 1, 1)).T
            # Cold Start: Initialize control trajectory with zero control
            U_guess = np.zeros((2, self.N))
        else:
            # Warm Start (CRITICAL FIX): Shift previous solution forward and append 
            # the last point as the new final state/control guess
            X_guess = self.X_opt
            U_guess = self.U_opt
            
        x0_guess = ca.vertcat(ca.vec(X_guess), ca.vec(U_guess))

        # --- 3. Solve NLP ---
        sol = self.solver(
            x0=x0_guess,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            p=P_val
        )
        
        # --- 4. Extract Solution ---
        x_opt = sol['x'].full().flatten()
        
        # Extract the state and control trajectories for warm-start in next step
        X_opt_next = np.reshape(x_opt[0:6 * (self.N + 1)], (6, self.N + 1))
        U_opt_next = np.reshape(x_opt[6 * (self.N + 1):], (2, self.N))
        
        # Store for Warm-Start (Shift forward)
        self.X_opt = np.hstack((X_opt_next[:, 1:], np.expand_dims(X_opt_next[:, -1], axis=1)))
        self.U_opt = np.hstack((U_opt_next[:, 1:], np.expand_dims(U_opt_next[:, -1], axis=1)))

        # Return the first control input
        U_next = U_opt_next[:, 0]
        return U_next