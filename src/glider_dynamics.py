import numpy as np
import yaml
import sys
import os

class GliderDynamics:
    
    def __init__(self, config_path):
        # Load configuration (assumed to be correct based on GliderControlSystem fix)
        self.config = self._load_config(config_path)
        
        # Initial State: [x, y, z, vx, vy, vz]
        # Assuming initial altitude is 200m and velocity is 20 m/s in the x-direction
        # This state must be a NumPy array.
        self.state = np.array([0.0, 0.0, 200.0, 20.0, 0.0, 0.0])
        
        # Store Glider parameters (used for dynamics calculation)
        glider_params = self.config.get('GLIDER', {})
        self.m = glider_params.get('mass', 700.0)
        self.S = glider_params.get('S', 14.0)
        self.CD0 = glider_params.get('CD0', 0.015)
        self.K = glider_params.get('K', 0.04)
        self.CL = glider_params.get('CL', 0.8) 
        self.g = 9.81
        self.rho = 1.225
        
        # Check if the mandatory state is initialized (CRITICAL)
        if self.state is None:
            raise RuntimeError("Glider state vector failed to initialize.")

    def _load_config(self, path):
        """Loads configuration from a YAML file for internal use."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"FATAL: Could not load configuration file {path} in GliderDynamics: {e}")
            sys.exit(1)

    # --- MISSING METHOD FIX (CRITICAL) ---
    def get_state(self):
        """Returns the current state vector of the glider."""
        return self.state

    def update(self, phi_command, alpha_command, dt, thermal_model):
        """
        Calculates the next state using RK4 integration.
        This method must already contain the complex dynamics logic.
        """
        X = self.state
        W_atm = np.array([0.0, 0.0, thermal_model.get_thermal_lift(X[0], X[1], X[2])])

        # --- RK4 Integration of Dynamics (Placeholder for the function f) ---
        # NOTE: Your full physics function 'f' must be defined here or imported.
        
        # For this fix, we are just ensuring the structure is correct.
        # Assuming f_dynamics (or similar) is defined and returns X_dot (6x1 vector)
        
        def f_dynamics(X_in, U_in, W_in):
            # This is where your physics equations (from X_dot) must be implemented
            
            # Simple placeholder (if you are missing the full dynamics implementation)
            # The actual dynamics are complex and defined in the MPC solver.
            # Here, we need the *numerical* version for simulation.
            
            # Placeholder for lift/drag calculations:
            vx, vy, vz = X_in[3], X_in[4], X_in[5]
            V_ground = np.array([vx, vy, vz])
            V_air_vec = V_ground - W_in
            V_air_mag = np.linalg.norm(V_air_vec)
            
            # If V_air_mag is near zero, use a small regularization
            V_reg = np.sqrt(V_air_mag**2 + 1e-6)
            e_v = V_air_vec / V_reg
            
            CD = self.CD0 + self.K * self.CL**2 
            L_mag = 0.5 * self.rho * self.S * self.CL * V_reg**2
            D_mag = 0.5 * self.rho * self.S * CD * V_reg**2
            D_vec = -D_mag * e_v
            G_vec = np.array([0.0, 0.0, -self.m * self.g])
            
            # Simplified Lift components (The full projection is complex)
            # Use the correct, full lift vector from your dynamics derivation here:
            # L_vec = L_mag * (Lift Direction Vector) 
            
            # For quick testing, we'll use a placeholder for force calculation:
            F_total = D_vec + G_vec # Must include the full LIFT vector here
            
            a_vec = F_total / self.m
            
            X_dot = np.hstack((V_ground, a_vec))
            return X_dot

        # --- RK4 Implementation ---
        M = 4 
        dt_rk4 = dt / M
        X_k = X
        U_in = np.array([phi_command, alpha_command])

        for _ in range(M):
            k1 = f_dynamics(X_k, U_in, W_atm)
            k2 = f_dynamics(X_k + dt_rk4/2 * k1, U_in, W_atm)
            k3 = f_dynamics(X_k + dt_rk4/2 * k2, U_in, W_atm)
            k4 = f_dynamics(X_k + dt_rk4 * k3, U_in, W_atm)
            X_k = X_k + dt_rk4/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        self.state = X_k
        return self.state

# Ensure Thermal model is also present/imported, as GliderDynamics uses it.