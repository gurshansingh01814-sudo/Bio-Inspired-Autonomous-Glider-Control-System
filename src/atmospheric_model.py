import numpy as np
import yaml
import os
import sys

class AtmosphericModel:
    
    def __init__(self, config_path):
        """Initializes atmospheric parameters and thermal properties."""
        
        self.config = self._load_config(config_path)
        
        # Load Thermal parameters
        thermal_params = self.config.get('THERMAL', {})
        self.thermal_center = np.array(thermal_params.get('center', [700.0, 0.0])) 
        self.thermal_radius = thermal_params.get('radius', 100.0) 
        self.W_z_max = thermal_params.get('max_lift', 4.0) 
        
        # Load other environment params
        env_params = self.config.get('ATMOSPHERE', {})
        self.rho = env_params.get('rho_air', 1.225)
        
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

    # --- CRITICAL FIX: Add the missing getter method ---
    def get_thermal_center(self):
        """Returns the [x, y] coordinates of the thermal center."""
        return self.thermal_center
    # ----------------------------------------------------

    def get_thermal_lift(self, x, y, z):
        """
        Calculates the vertical wind (Wz) at a given (x, y, z) location.
        This simplified model returns 0 outside the thermal radius, and W_z_max inside.
        (A real model would use a smooth distribution).
        """
        # Calculate horizontal distance to the thermal center
        dx = x - self.thermal_center[0]
        dy = y - self.thermal_center[1]
        horizontal_distance = np.sqrt(dx**2 + dy**2)
        
        if horizontal_distance <= self.thermal_radius:
            # Simple assumption: Full max lift inside the cylinder
            return self.W_z_max
        else:
            return 0.0 # No lift outside