
import numpy as np
import yaml
import os

class AtmosphericModel:
    """
    Loads configuration and provides simplified thermal location and wind vector for simulation.
    """
    def __init__(self, full_config_path):
        # Even if this class doesn't use the config for now, it must accept the path input.
        
        # Thermal parameters (hardcoded for now, but ready for config integration)
        self.thermal_cx = 100.0  # Center X
        self.thermal_cy = 100.0  # Center Y
        self.thermal_radius = 200.0
        self.W_z_max = 3.0
        
    def get_thermal_center(self):
        return np.array([self.thermal_cx, self.thermal_cy])

    def get_thermal_radius(self):
        return self.thermal_radius
        
    def get_wind_vector(self, position):
        """
        Calculates the wind vector [wx, wy, wz] at a given position [x, y, z].
        Only vertical wind (wz) is non-zero, simulating a thermal.
        """
        x, y, z = position
        dist = np.sqrt((x - self.thermal_cx)**2 + (y - self.thermal_cy)**2)
        
        W_z = 0.0
        if dist < self.thermal_radius:
            dist_ratio = dist / self.thermal_radius
            # Smooth cosine model for uplift
            W_z = self.W_z_max * (np.cos(np.pi * dist_ratio) + 1.0) / 2.0
            
        return np.array([0.0, 0.0, W_z])

    def get_thermal_state(self, position):
        """Returns True if the glider is inside the thermal radius."""
        x, y, z = position
        dist = np.sqrt((x - self.thermal_cx)**2 + (y - self.thermal_cy)**2)
        return dist < self.thermal_radius
