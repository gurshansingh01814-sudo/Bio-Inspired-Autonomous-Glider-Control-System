import numpy as np
import yaml
import os
import sys

class GliderDynamics:
    # ... (init method remains the same) ...
    def __init__(self, full_config_path):
        # --- Config Loading (Now Robust) ---
        config = {}
        try:
            with open(full_config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"FATAL: YAML parsing error in GliderDynamics: {e}")
            sys.exit(1)

        # --- Glider Parameters ---
        params = config.get('GLIDER', {}) 
        self.m = params.get('mass', 700.0)
        self.S = params.get('S', 14.0)
        self.CD0 = params.get('CD0', 0.015)
        self.K = params.get('K', 0.04)
        self.CL = params.get('CL', 0.8)
        
        self.g = 9.81
        self.rho = 1.225
        self.EPSILON_AIRSPEED = 1e-6 # Must match MPC for consistency


    def step(self, Xk, phi, alpha, W_atm, dt):
        """
        FIXED: Aero-based dynamics model (must match MPC structure for feasibility).
        Xk = [x, y, z, vx, vy, vz]
        Control: phi (bank), alpha (effective pitch)
        W_atm: Atmospheric wind vector [wx, wy, wz]
        """
        
        # --- Unpack State and Control ---
        x, y, z, vx, vy, vz = Xk
        V_ground_vec = np.array([vx, vy, vz])

        # --- Airspeed and Unit Vector (V_air) ---
        V_air_vec = V_ground_vec - W_atm
        V_air = np.linalg.norm(V_air_vec)

        # CRITICAL: Use the regularization term to stabilize V_air if needed
        V_reg = np.sqrt(V_air**2 + self.EPSILON_AIRSPEED)
        e_v = V_air_vec / V_reg if V_reg > 0 else np.zeros(3) # Unit vector of V_air

        # --- Aerodynamic Forces ---
        CD = self.CD0 + self.K * self.CL**2 # Simplified constant drag
        
        L_mag = 0.5 * self.rho * self.S * self.CL * V_reg**2
        D_mag = 0.5 * self.rho * self.S * CD * V_reg**2
        
        D_vec = -D_mag * e_v
        
        # Lift Force simplified projection (using control alpha/phi)
        L_x = L_mag * np.sin(phi) * np.sin(alpha) 
        L_y = -L_mag * np.cos(phi) * np.sin(alpha) 
        L_z = L_mag * np.cos(alpha) 
        L_vec = np.array([L_x, L_y, L_z])

        # --- Total Force & Acceleration ---
        G_vec = np.array([0.0, 0.0, -self.m * self.g])
        F_total = L_vec + D_vec + G_vec
        
        # Acceleration is F/m
        a_vec = F_total / self.m
        
        X_dot = np.array([vx, vy, vz, a_vec[0], a_vec[1], a_vec[2]])
        
        # Euler integration step
        Xk_next = Xk + dt * X_dot
        
        # Safety Check: Limit altitude
        if Xk_next[2] < 5.0:
            Xk_next[2] = 5.0
            Xk_next[5] = 0.0
            
        return Xk_next

