import numpy as np
import math

class GliderDynamics:
    """
    Simulates the non-linear dynamics of a glider, treating it as a point mass.
    
    State vector X = [x, y, z, vx, vy, vz, m]
    Control input U = [phi, gamma] (bank angle, pitch/AoA proxy)
    """
    def __init__(self, params):
        # Glider physical parameters loaded from main_stimulation.py
        self.m = params.get('mass', 700.0)
        self.g = 9.81
        self.rho = 1.225 # Air density (kg/m^3)
        self.S = params.get('S', 14.0) # Wing area (m^2)
        self.CD0 = params.get('CD0', 0.015) # Zero-lift drag coefficient
        self.K = params.get('K', 0.04) # Induced drag factor
        self.CL = 0.8 # Constant Lift coefficient placeholder (simplified for MPC consistency)

    def calculate_airspeed(self, state, wind_vector):
        """
        Calculates the magnitude of the air velocity vector V_air, which is 
        V_air = V_ground - W_atm.
        """
        V_ground = state[3:6]
        V_air_vec = V_ground - wind_vector
        return np.linalg.norm(V_air_vec)

    def step_dynamics(self, state, phi, gamma, wind_vector, dt):
        """
        Integrates the dynamics over a time step dt using Euler integration.
        
        Args:
            state (np.array): Current state vector [x, y, z, vx, vy, vz, m].
            phi (float): Bank angle (rad).
            gamma (float): Pitch angle (rad), used as a proxy for Angle of Attack control.
            wind_vector (np.array): Atmospheric wind vector [Wx, Wy, Wz].
            dt (float): Time step duration.
            
        Returns:
            np.array: The state vector at the next time step.
        """
        x, y, z, vx, vy, vz, m = state
        
        # 1. Calculate Airspeed and Direction
        V_ground_vec = np.array([vx, vy, vz])
        V_air_vec = V_ground_vec - wind_vector
        V_air_mag = np.linalg.norm(V_air_vec)
        
        # Stability check: ensure we don't divide by zero if speed is too low
        if V_air_mag < 0.1: 
            V_air_mag = 0.1
            
        e_v = V_air_vec / V_air_mag # Unit vector of V_air (direction of motion through air)
        
        # 2. Aerodynamic Forces
        # Drag coefficient based on simplified parabolic drag polar
        CD = self.CD0 + self.K * self.CL**2
        
        L = 0.5 * self.rho * self.S * self.CL * V_air_mag**2 # Lift magnitude
        D = 0.5 * self.rho * self.S * CD * V_air_mag**2     # Drag magnitude
        
        # Drag acts opposite to V_air
        D_vec = -D * e_v
        
        # Simplified Lift projection (aligned with control commands)
        # This projection relates lift to the body frame defined by bank (phi) and pitch (gamma)
        L_x = L * (math.sin(phi) * math.sin(gamma)) 
        L_y = L * (-math.cos(phi) * math.sin(gamma)) 
        L_z = L * (math.cos(gamma)) 
        L_vec = np.array([L_x, L_y, L_z])
        
        # 3. Total Force and Acceleration
        G_vec = np.array([0.0, 0.0, -m * self.g]) # Gravity
        F_total = L_vec + D_vec + G_vec
        a_vec = F_total / m # Acceleration = Force / Mass
        
        # 4. State Derivatives and Integration (Euler step)
        # x_dot = [vx, vy, vz, ax, ay, az, 0]
        x_dot = np.concatenate([V_ground_vec, a_vec, [0.0]]) 
        next_state = state + dt * x_dot
        
        # Ensure mass remains constant (the last element of the state vector)
        next_state[6] = self.m 

        return next_state
