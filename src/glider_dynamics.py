import numpy as np
import math

class GliderDynamics:
    """
    Simulates the non-linear dynamics of a glider.
    Uses a standard lift, drag, and gravity model.
    """
    def __init__(self, params):
        self.m = params.get('mass', 700.0)
        self.g = 9.81
        self.rho = 1.225 # Air density (kg/m^3)
        self.S = params.get('S', 14.0) # Wing area (m^2)
        self.CD0 = params.get('CD0', 0.015) # Zero-lift drag coefficient
        self.K = params.get('K', 0.04) # Induced drag factor (1 / (pi * e * AR))
        self.CL = 0.8 # Constant Lift coefficient placeholder (A standard simplification)

    def calculate_airspeed(self, state, wind_vector):
        """Calculates the magnitude of the air velocity vector."""
        # state = [x, y, z, vx, vy, vz, m]
        V_ground = state[3:6]
        V_air_vec = V_ground - wind_vector
        return np.linalg.norm(V_air_vec)

    def step_dynamics(self, state, phi, gamma, wind_vector, dt):
        """
        Integrates the dynamics over a time step dt using Euler integration.
        phi (rad): bank angle
        gamma (rad): pitch angle (proxy for angle of attack control)
        """
        x, y, z, vx, vy, vz, m = state
        
        # 1. Airspeed and Aerodynamics
        V_ground_vec = np.array([vx, vy, vz])
        V_air_vec = V_ground_vec - wind_vector
        V_air_mag = np.linalg.norm(V_air_vec)
        
        if V_air_mag < 0.1: # Prevent division by zero or numerical instability
            V_air_mag = 0.1
            
        e_v = V_air_vec / V_air_mag # Unit vector of V_air
        
        # Total Drag Coefficient
        CD = self.CD0 + self.K * self.CL**2
        
        # Force Magnitudes
        L = 0.5 * self.rho * self.S * self.CL * V_air_mag**2
        D = 0.5 * self.rho * self.S * CD * V_air_mag**2
        
        # 2. Force Vectors
        
        # Drag acts opposite to V_air
        D_vec = -D * e_v
        
        # Gravity
        G_vec = np.array([0.0, 0.0, -m * self.g])
        
        # Simplified Lift projection (aligned with the CasADi model)
        L_x = L * (math.sin(phi) * math.sin(gamma)) 
        L_y = L * (-math.cos(phi) * math.sin(gamma)) 
        L_z = L * (math.cos(gamma)) 
        L_vec = np.array([L_x, L_y, L_z])
        
        # 3. Total Force and Acceleration
        F_total = L_vec + D_vec + G_vec
        a_vec = F_total / m
        
        # 4. State Derivatives (dx/dt)
        # Derivatives: [vx, vy, vz, ax, ay, az, 0]
        x_dot = np.concatenate([V_ground_vec, a_vec, [0.0]]) 
        
        # 5. Euler Integration
        next_state = state + dt * x_dot
        next_state[6] = self.m # Mass remains constant

        return next_state