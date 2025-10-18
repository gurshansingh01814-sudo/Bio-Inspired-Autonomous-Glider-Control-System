import numpy as np
import yaml
import sys
import os
from numpy.linalg import norm 

# Helper: Converts degrees to radians for internal calculations
def deg_to_rad(deg):
    return deg * (np.pi / 180.0)

class GliderDynamics:
    
    def __init__(self, config_path):
        """Initializes the glider state and loads physical parameters from config."""
        
        self.config = self._load_config(config_path)
        
        # Load SIMULATION parameters
        sim_params = self.config.get('SIMULATION', {})
        # Assuming initial state is X = [x, y, z, vx, vy, vz]
        self.state = np.array(sim_params.get('initial_state', [0.0, 0.0, 400.0, 20.0, 0.0, 0.0]))
        
        # Load GLIDER parameters
        glider_params = self.config.get('GLIDER', {})
        self.m = glider_params.get('mass', 20.0)        # Mass (kg)
        self.S = glider_params.get('S', 14.0)          # Wing area (m^2)
        self.CD0 = glider_params.get('CD0', 0.015)     # Zero-lift drag coefficient
        self.K = glider_params.get('K', 0.04)          # Induced drag factor
        
        # Load ATMOSPHERE parameters
        env_params = self.config.get('ATMOSPHERE', {})
        self.g = env_params.get('gravity', 9.81)       # Gravitational acceleration
        self.rho = env_params.get('rho_air', 1.225)    # Air density
        
        # CRITICAL FIX: Define EPSILON_LIFT here (matching the symbolic MPC stability fix)
        self.EPSILON_AIRSPEED = 1e-6                   # Regularization for division by zero (air speed)
        self.EPSILON_LIFT = 1e-4                       # Robust regularization for lift vector calculation

    def _load_config(self, path):
        """Loads configuration from a YAML file. Simplified to a dictionary."""
        if not os.path.exists(path):
            print(f"FATAL: Configuration file not found at {path}")
            sys.exit(1)
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"FATAL: Could not load configuration file {path}: {e}")
            sys.exit(1)

    def get_state(self):
        """Returns the current state vector of the glider."""
        return self.state

    def update(self, CL_command, phi_command, dt, atmospheric_model):
        """
        Calculates the next state using RK4 integration over the timestep dt.
        Control Inputs (U_in):
        - CL_command: Commanded Lift Coefficient (u1)
        - phi_command: Commanded Bank Angle (u2)
        """
        X = self.state
        
        # Get atmospheric vertical wind (Wz) from the thermal model
        W_atm_z = atmospheric_model.get_thermal_lift(X[0], X[1], X[2])
        W_atm = np.array([0.0, 0.0, W_atm_z])
        U_in = np.array([CL_command, phi_command]) # Note: phi must be in radians here

        def f_dynamics(X_in, U_in, W_in):
            """Returns the state derivatives (X_dot = [V, a]) for RK4."""
            
            # State and Control
            vx, vy, vz = X_in[3], X_in[4], X_in[5]
            CL, phi = U_in[0], U_in[1]
            
            V_ground = np.array([vx, vy, vz])
            
            # Air velocity calculation (V_air = V_ground - W_wind)
            V_air_vec = V_ground - W_in
            V_air_mag = norm(V_air_vec)
            V_reg = np.sqrt(V_air_mag**2 + self.EPSILON_AIRSPEED)
            e_v = V_air_vec / V_reg # Unit vector in direction of air velocity
            
            # 1. Drag Force (Fd)
            CD = self.CD0 + self.K * CL**2              # Induced drag depends on current CL
            D_mag = 0.5 * self.rho * self.S * CD * V_reg**2
            F_drag = -D_mag * e_v                       # Drag opposes air velocity vector
            
            # 2. Lift Force (Fl)
            L_mag = 0.5 * self.rho * self.S * CL * V_reg**2
            
            # Project Z-axis onto the plane perpendicular to e_v
            e_z = np.array([0.0, 0.0, 1.0])
            e_L_raw = e_z - np.dot(e_z, e_v) * e_v 
            
            # FIX: Use the defined self.EPSILON_LIFT for robust normalization
            L_vert_unit = e_L_raw / (norm(e_L_raw) + self.EPSILON_LIFT)
            
            # The cross-product gives the vector perpendicular to both L_vert_unit and e_v
            L_side_unit = np.cross(e_v, L_vert_unit) 

            # Lift Vector: Components are rotated by the bank angle (phi)
            F_lift = L_mag * (np.cos(phi) * L_vert_unit + np.sin(phi) * L_side_unit)

            # 3. Gravity Force (Fg)
            F_gravity = np.array([0.0, 0.0, -self.m * self.g])
            
            # Total Force and Acceleration (F = ma)
            F_total = F_lift + F_drag + F_gravity
            a_vec = F_total / self.m
            
            # X_dot = [V_ground, a_vec]
            X_dot = np.hstack((V_ground, a_vec))
            return X_dot

        # --- RK4 Integration (Standard Implementation) ---
        M = 4 # Number of RK4 steps per simulation step
        dt_rk4 = dt / M
        X_k = X
        
        for _ in range(M):
            k1 = f_dynamics(X_k, U_in, W_atm)
            k2 = f_dynamics(X_k + dt_rk4/2 * k1, U_in, W_atm)
            k3 = f_dynamics(X_k + dt_rk4/2 * k2, U_in, W_atm)
            k4 = f_dynamics(X_k + dt_rk4 * k3, U_in, W_atm)
            X_k = X_k + dt_rk4/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        self.state = X_k
        return self.state