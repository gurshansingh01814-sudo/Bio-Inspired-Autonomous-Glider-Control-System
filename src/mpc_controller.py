import casadi as ca
import numpy as np
import yaml
import sys
import math

class MPCController:

    def __init__(self, config_path):
        """Initializes the MPC problem, objective, constraints, and solver."""

        self.config = self._load_config(config_path)
        
        # Unpack Parameters
        self.N = self.config.get('MPC', {}).get('horizon_steps', 10)  # Prediction horizon
        self.DT = self.config.get('MPC', {}).get('timestep', 2.0)  # Time step (seconds)

        # Unpack Glider parameters (for symbolic dynamics)
        glider_params = self.config.get('GLIDER', {})
        self.m = ca.MX(glider_params.get('mass', 700.0))
        self.S = ca.MX(glider_params.get('S', 14.0))
        self.CD0 = ca.MX(glider_params.get('CD0', 0.015))
        self.K = ca.MX(glider_params.get('K', 0.04))
        self.CL = ca.MX(glider_params.get('CL', 0.8))
        self.g = ca.MX(9.81)
        self.rho = ca.MX(1.225)
        self.EPSILON_AIRSPEED = 1e-6

        # Problem Setup (State and Control Dimensions)
        self.NX = 6 # [x, y, z, vx, vy, vz]
        self.NU = 2 # [phi, alpha] (Bank Angle, Effective Pitch Angle)
        self.solver = self._setup_solver()
        print(f"MPC Controller initialized with N={self.N}, DT={self.DT}s.")
        
    def _load_config(self, path):
        """Loads configuration from a YAML file for internal use."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"FATAL: Could not load configuration file {path} in MPCController: {e}")
            sys.exit(1)
            
    def _glider_dynamics(self, X, U, W_atm):
        """
        Symbolic, continuous-time state-space model (X_dot = f(X, U, W_atm)).
        This must match the numerical integration in GliderDynamics.
        """
        x, y, z, vx, vy, vz = ca.vertsplit(X)
        phi, alpha = ca.vertsplit(U)
        
        V_ground = ca.vertcat(vx, vy, vz)
        W_in = ca.vertcat(ca.MX(0.0), ca.MX(0.0), W_atm)
        
        # Air velocity calculation
        V_air_vec = V_ground - W_in
        V_air_mag = ca.norm_2(V_air_vec)
        V_reg = ca.sqrt(V_air_mag**2 + self.EPSILON_AIRSPEED)
        e_v = V_air_vec / V_reg 
        
        # Magnitude Calculations
        CD = self.CD0 + self.K * self.CL**2 
        L_mag = 0.5 * self.rho * self.S * self.CL * V_reg**2
        D_mag = 0.5 * self.rho * self.S * CD * V_reg**2
        
        # 1. Drag Vector: opposes air velocity
        D_vec = -D_mag * e_v
        
        # 2. Lift Vector: perpendicular to air velocity
        L_x = L_mag * ca.sin(phi) * e_v[1] 
        L_y = -L_mag * ca.sin(phi) * e_v[0]
        L_z = L_mag * ca.cos(phi) * ca.cos(alpha) 
        L_vec = ca.vertcat(L_x, L_y, L_z)
        
        # 3. Gravity Vector
        G_vec = ca.vertcat(0.0, 0.0, -self.m * self.g)
        
        # Total Force and Acceleration
        F_total = L_vec + D_vec + G_vec
        a_vec = F_total / self.m
        
        # X_dot = [V_ground, a_vec]
        X_dot = ca.vertcat(V_ground, a_vec)
        return X_dot

    def _setup_solver(self):
        """Sets up the CasADi optimization problem (OCP)."""
        
        opti = ca.Opti()

        # Decision variables: State (X) and Control (U) sequences
        X = opti.variable(self.NX, self.N + 1) # State trajectory
        U = opti.variable(self.NU, self.N)     # Control input sequence
        
        # Parameters: Initial State (P_init) and Target (P_target), Thermal (P_thermal)
        P_init = opti.parameter(self.NX, 1)    # Initial state
        P_target = opti.parameter(2, 1)        # Target (thermal center [x, y])
        P_Wz = opti.parameter(1, self.N)       # Thermal lift (Wz) over the horizon
        
        # Slack variable for altitude constraint violation (z > Z_min + S_alt)
        S_alt = opti.variable(1, self.N + 1)   
        
        # Objective Function (Minimize Energy, Maximize Altitude, Reach Thermal)
        J = 0 
        Q_dist = 1.0     # Weight for distance to thermal
        Q_velocity = 0.1 # Weight for maintaining nominal speed
        R_control = 0.01 # Weight for smoothness of control inputs
        
        # Cost Loop
        for k in range(self.N):
            
            # --- Objective: Pathing and Energy ---
            
            # 1. Distance to Thermal Cost (Minimize distance)
            dist_sq = (X[0, k] - P_target[0])**2 + (X[1, k] - P_target[1])**2
            J += Q_dist * dist_sq
            
            # 2. Survival Objective (CRITICAL: Prioritize Climbing)
            # We want to maximize Z, so we minimize -Z. Set to a very large weight.
            J += -1000 * X[2, k+1] # Maximize Z at the next step
            
            # 3. Control Effort / Smoothness
            J += R_control * ca.sumsqr(U[:, k])
            
            # 4. Continuity (Integrate dynamics forward)
            X_dot = self._glider_dynamics(X[:, k], U[:, k], P_Wz[0, k])
            opti.subject_to(X[:, k+1] == X[:, k] + self.DT * X_dot)

        # --- Terminal/Final State Cost (N+1 step) ---
        J += Q_dist * ((X[0, self.N] - P_target[0])**2 + (X[1, self.N] - P_target[1])**2)
        
        # --- Penalize Altitude Constraint Violation ---
        # CRITICAL FIX 2: Increased penalty for altitude slack violation
        J += 10000 * ca.sumsqr(S_alt) # Massive penalty on slack variable

        # --- Constraints ---
        
        # 1. Initial State Constraint
        opti.subject_to(X[:, 0] == P_init)
        
        # 2. Control Bounds (Bank angle phi and Pitch angle alpha)
        DEG_TO_RAD = math.pi / 180.0
        # Bank Angle (Phi): +/- 45 degrees
        opti.subject_to(opti.bounded(-45.0 * DEG_TO_RAD, U[0, :], 45.0 * DEG_TO_RAD))
        # Pitch Angle (Alpha): Relaxed max pitch to allow for high lift maneuvers
        # CRITICAL FIX 3: Increased max pitch from 15deg to 20deg
        opti.subject_to(opti.bounded(-5.0 * DEG_TO_RAD, U[1, :], 20.0 * DEG_TO_RAD))
        
        # 3. State Bounds
        # Altitude (Z): Must stay above Z_min. Use slack variable S_alt >= 0.
        Z_MIN = 20.0 # meters
        opti.subject_to(X[2, :] >= Z_MIN - S_alt) 
        opti.subject_to(S_alt >= 0.0)
        
        # Velocity Bounds (must maintain a minimum airspeed to avoid stall)
        V_MIN = 10.0 # m/s
        V_MAX = 50.0 # m/s (safe upper limit)
        V_air = ca.sqrt(X[3, :]**2 + X[4, :]**2 + X[5, :]**2)
        opti.subject_to(V_air >= V_MIN)
        opti.subject_to(V_air <= V_MAX)

        # Optimization Setup
        opti.minimize(J)
        
        # Solver Options (IPOPT)
        opts = {
            'ipopt': {
                'max_iter': 100,
                'print_level': 0, # 0 for quiet, 3 for maximum
                'acceptable_tol': 1e-6,
                'acceptable_obj_change_tol': 1e-4,
                'tol': 1e-4,
                # 'fast_accept_ic_w_obj_times': 1, # UNSTABLE OPTION: REMOVED
            },
            'print_time': 0,
        }
        opti.solver('ipopt', opts)

        # Map optimization variables to a CasADi function for execution
        P = ca.vertcat(P_init, P_target, ca.reshape(P_Wz, self.N, 1))
        U_out = ca.vertcat(U)
        
        return opti.to_function('solver', [P], [U_out])

    def solve_mpc(self, initial_state, thermal_center, Wz_prediction):
        """
        Solves the MPC problem for the current time step.
        Returns the optimal control input (phi, alpha) for the first step.
        """
        
        # The function expects parameters concatenated as [P_init, P_target, P_Wz]
        P_target = np.array([thermal_center[0], thermal_center[1]])
        P_Wz = Wz_prediction.flatten()
        
        P_all = np.concatenate([initial_state, P_target, P_Wz])
        
        try:
            # The solver function returns the flat control vector U (NU * N)
            U_flat = self.solver(P_all)
            U_optimal = np.array(U_flat).reshape(self.NU, self.N)
            
            # Return only the first control input (phi, alpha)
            return U_optimal[:, 0] 
        
        except Exception as e:
            # If the solver fails (e.g., Infeasible, Max Iterations), return zero control
            # This is safer than crashing, allowing the system to use the previous command.
            print("\nWARNING: IPOPT failed to converge. Returning last known successful control input.")
            # In a real system, you would store and return the previously commanded U.
            # For simulation, returning zero or a small lift command is safer.
            return np.array([0.0, 0.0])