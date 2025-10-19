import numpy as np
import yaml
import os
import sys
import math

class GliderDynamics:
    """
    Implements the NUMERICAL, continuous-time state-space model for the glider 
    used for simulation (integration) over the small time step (dt_sim).
    """

    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        
        # Unpack Glider parameters
        glider_params = self.config.get('GLIDER', {})
        env_params = self.config.get('ATMOSPHERE', {})
        self.m = glider_params.get('mass', 20.0)
        self.S = glider_params.get('S', 14.0)
        self.CD0 = glider_params.get('CD0', 0.015)
        self.K = glider_params.get('K', 0.04)
        self.g = env_params.get('gravity', 9.81)
        self.rho = env_params.get('rho_air', 1.225)
        
        # Initial State: [x, y, z, vx, vy, vz]
        self.X = np.array(glider_params.get('initial_state', [0.0, 0.0, 500.0, 15.0, 0.0, -1.0]))
        
        # Numerical stability constants
        self.EPSILON_AIRSPEED = 1e-4 
        self.EPSILON_LIFT = 1e-3     
        
        self.CL = 0.2  
        self.phi = 0.0 
        
    def _load_config(self, path):
        """Loads configuration from a YAML file for internal use."""
        if not os.path.exists(path):
            print(f"FATAL: Configuration file not found at {path}")
            sys.exit(1)
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"FATAL: Could not load configuration file {path}: {e}")
            sys.exit(1)

    def _state_dot(self, X, CL, phi, W_atm_z):
        """
        Numerical, continuous-time state-space model (X_dot = f(X, U, W_atm)).
        Uses NumPy for calculations.
        """
        X = np.asarray(X)
        vx, vy, vz = X[3], X[4], X[5]
        
        V_ground = np.array([vx, vy, vz])
        W_in = np.array([0.0, 0.0, W_atm_z])
        
        # Air velocity calculation
        V_air_vec = V_ground - W_in
        V_air_mag = np.linalg.norm(V_air_vec)
        
        # 1. Airspeed Regularization (V_reg)
        V_reg = np.sqrt(V_air_mag**2 + self.EPSILON_AIRSPEED**2) 
        e_v = V_air_vec / V_reg 
        
        # 1. Drag Force (Fd)
        CD = self.CD0 + self.K * CL**2              
        D_mag = 0.5 * self.rho * self.S * CD * V_reg**2
        F_drag = -D_mag * e_v                       
        
        # 2. Lift Force (Fl)
        L_mag = 0.5 * self.rho * self.S * CL * V_reg**2
        
        e_z = np.array([0.0, 0.0, 1.0])
        
        # Projection and perpendicular vectors
        dot_product = np.dot(e_z, e_v)
        e_L_raw = e_z - dot_product * e_v 
        
        # 2. Lift Vector Regularization
        L_raw_mag_sq_reg = np.sum(e_L_raw**2) + self.EPSILON_LIFT**2
        L_raw_mag_reg = np.sqrt(L_raw_mag_sq_reg)

        L_vert_unit = e_L_raw / L_raw_mag_reg
        
        L_side_unit = np.cross(e_v, L_vert_unit) 

        # Lift Vector: Components are rotated by the bank angle (phi)
        F_lift = L_mag * (np.cos(phi) * L_vert_unit + np.sin(phi) * L_side_unit)

        # 3. Gravity Force (Fg)
        F_gravity = np.array([0.0, 0.0, -self.m * self.g])
        
        # Total Force and Acceleration (F = ma)
        F_total = F_lift + F_drag + F_gravity
        a_vec = F_total / self.m
        
        # X_dot = [V_ground, a_vec]
        X_dot = np.concatenate([V_ground, a_vec])
        return X_dot

    def update(self, CL_cmd, phi_cmd, dt_sim, atmospheric_model):
        """
        Integrates the glider dynamics forward one time step (dt_sim) using 
        Fourth-Order Runge-Kutta (RK4) integration.
        """
        self.CL = CL_cmd
        self.phi = phi_cmd

        # Get atmospheric lift at the current glider position
        x, y, z = self.X[0], self.X[1], self.X[2]
        W_atm_z = atmospheric_model.get_thermal_lift(x, y, z)
        
        # RK4 Integration Steps
        
        # K1: Start slope
        K1 = self._state_dot(self.X, CL_cmd, phi_cmd, W_atm_z)
        
        # K2: Midpoint slope 1
        X2 = self.X + (dt_sim / 2.0) * K1
        K2 = self._state_dot(X2, CL_cmd, phi_cmd, W_atm_z)
        
        # K3: Midpoint slope 2
        X3 = self.X + (dt_sim / 2.0) * K2
        K3 = self._state_dot(X3, CL_cmd, phi_cmd, W_atm_z)
        
        # K4: End slope
        X4 = self.X + dt_sim * K3
        K4 = self._state_dot(X4, CL_cmd, phi_cmd, W_atm_z)
        
        # Final state update: X_new = X_old + dt/6 * (K1 + 2*K2 + 2*K3 + K4)
        self.X = self.X + (dt_sim / 6.0) * (K1 + 2.0 * K2 + 2.0 * K3 + K4)
        
        # Return the new state
        return self.X

    def get_state(self):
        """Returns the current state of the glider: [x, y, z, vx, vy, vz]."""