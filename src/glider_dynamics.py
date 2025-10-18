import numpy as np
import yaml
import os
import sys

class GliderDynamics:
    """
    Implements the simplified glider dynamics model and loads parameters from config.
    """
    def __init__(self, full_config_path):
        
        # --- Config Loading (Now Robust) ---
        config = {}
        try:
            with open(full_config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            # Since main_simulation checks file existence, this handles parsing error
            print(f"FATAL: YAML parsing error in GliderDynamics: {e}")
            sys.exit(1)

        # --- Glider Parameters ---
        # The core parameters dictionary is now GLIDER
        params = config.get('GLIDER', {}) 
        
        # We are now calling .get() on the 'params' dictionary, not the full_config_path string!
        self.m = params.get('mass', 700.0)
        self.S = params.get('S', 14.0)
        self.CD0 = params.get('CD0', 0.015)
        self.K = params.get('K', 0.04)
        self.CL = params.get('CL', 0.8)
        
        # Gravity constant
        self.g = 9.81
        
        # Air density (constant for this simplified model)
        self.rho = 1.225 


    def step(self, Xk, phi, alpha, W_atm, dt):
        """
        Performs a simple state integration (placeholder physics).
        Xk = [x, y, z, vx, vy, vz]
        Control: phi (bank), alpha (effective pitch)
        W_atm: Atmospheric wind vector [wx, wy, wz]
        """
        
        # Unpack state
        x, y, z, vx, vy, vz = Xk
        
        # Calculate approximate V_air (simplistic)
        V_ground = np.array([vx, vy, vz])
        V_air_vec = V_ground - W_atm
        V_air = np.linalg.norm(V_air_vec)

        # Thrust/Lift placeholder (Simplistic: Vertical acceleration is proportional to alpha)
        
        # dx/dt = V_ground
        x_dot = vx
        y_dot = vy
        z_dot = vz
        
        # dvx/dt (Decoupled: small friction)
        vx_dot = -0.1 * vx 
        vy_dot = -0.1 * vy
        
        # dvz/dt (Gravity + Uplift + Control)
        # If V_air is low, glider stalls
        if V_air < 10.0:
            vz_dot = -self.g + W_atm[2] - 5.0 # Increased sink rate if stalling
        else:
            vz_dot = -self.g + W_atm[2] + np.rad2deg(alpha) * 0.1 # Simple vertical control via alpha

        X_dot = np.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot])
        
        # Euler integration step
        Xk_next = Xk + dt * X_dot
        
        # Ensure altitude constraint is met
        if Xk_next[2] < 0.0:
            Xk_next[2] = 0.0
            Xk_next[5] = 0.0 # Vertical speed is zeroed
            
        return Xk_next

