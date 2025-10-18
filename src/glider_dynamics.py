import numpy as np
import yaml
import sys
import os

class GliderDynamics:
    
    def __init__(self, config_path):
        """Initializes the glider state and loads physical parameters."""
        
        self.config = self._load_config(config_path)
        
        # Initial State: [x, y, z, vx, vy, vz]
        # Starting point and initial velocity (e.g., 20 m/s in the x-direction)
        self.state = np.array([0.0, 0.0, 200.0, 20.0, 0.0, 0.0])
        
        # Load Glider parameters
        glider_params = self.config.get('GLIDER', {})
        self.m = glider_params.get('mass', 700.0)  # Mass (kg)
        self.S = glider_params.get('S', 14.0)      # Wing area (m^2)
        self.CD0 = glider_params.get('CD0', 0.015) # Zero-lift drag coefficient
        self.K = glider_params.get('K', 0.04)      # Induced drag factor
        self.CL = glider_params.get('CL', 0.8)     # Lift coefficient (constant)
        
        # Environmental constants
        self.g = 9.81
        self.rho = 1.225
        self.EPSILON_AIRSPEED = 1e-6 # Regularization for division by zero

    def _load_config(self, path):
        """Loads configuration from a YAML file for internal use."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"FATAL: Could not load configuration file {path} in GliderDynamics: {e}")
            sys.exit(1)

    def get_state(self):
        """Returns the current state vector of the glider."""
        return self.state

    def update(self, phi_command, alpha_command, dt, atmospheric_model):
        """
        Calculates the next state using RK4 integration over the timestep dt.
        """
        X = self.state
        
        # Get atmospheric vertical wind (Wz) from the thermal model
        W_atm = np.array([0.0, 0.0, atmospheric_model.get_thermal_lift(X[0], X[1], X[2])])
        U_in = np.array([phi_command, alpha_command])

        def f_dynamics(X_in, U_in, W_in):
            """Returns the state derivatives (X_dot = [V, a]) for RK4."""
            
            # State and Control
            vx, vy, vz = X_in[3], X_in[4], X_in[5]
            phi, alpha = U_in[0], U_in[1]
            
            V_ground = np.array([vx, vy, vz])
            
            # Air velocity calculation
            V_air_vec = V_ground - W_in
            V_air_mag = np.linalg.norm(V_air_vec)
            V_reg = np.sqrt(V_air_mag**2 + self.EPSILON_AIRSPEED)
            e_v = V_air_vec / V_reg # Unit vector in direction of air velocity
            
            # Magnitude Calculations
            CD = self.CD0 + self.K * self.CL**2 
            L_mag = 0.5 * self.rho * self.S * self.CL * V_reg**2
            D_mag = 0.5 * self.rho * self.S * CD * V_reg**2
            
            # 1. Drag Vector: opposes air velocity
            D_vec = -D_mag * e_v
            
            # 2. Lift Vector: perpendicular to air velocity, tilted by control inputs
            if V_reg > 1e-3:
                # Lift components derived from the MPC formulation (CasADi)
                # These forces must counter Gravity and enable banking/turning.
                
                L_x = L_mag * np.sin(phi) * e_v[1] 
                L_y = -L_mag * np.sin(phi) * e_v[0]
                L_z = L_mag * np.cos(phi) * np.cos(alpha) 
                L_vec = np.array([L_x, L_y, L_z])
            else:
                 L_vec = np.zeros(3)
            
            # 3. Gravity Vector
            G_vec = np.array([0.0, 0.0, -self.m * self.g])
            
            # Total Force and Acceleration
            F_total = L_vec + D_vec + G_vec
            a_vec = F_total / self.m
            
            X_dot = np.hstack((V_ground, a_vec))
            return X_dot

        # --- RK4 Integration ---
        M = 4 
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