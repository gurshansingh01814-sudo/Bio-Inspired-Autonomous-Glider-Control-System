import numpy as np
import yaml
import os
import sys

class AtmosphericModel:
    
    def __init__(self, config_path):
        """Initializes atmospheric parameters and thermal properties."""
        
        self.config = self._load_config(config_path)
        
        # Load the main ATMOSPHERE section first
        env_params = self.config.get('ATMOSPHERE', {})
        
        # --- FIX 1: Access the nested THERMAL section correctly ---
        thermal_params = env_params.get('THERMAL', {})
        
        # --- FIX 2 & 3: Map config keys to instance attributes correctly ---
        # The thermal center is now constructed from the separate X and Y keys
        center_x = thermal_params.get('center_x', 500.0)
        center_y = thermal_params.get('center_y', 500.0)
        self.thermal_center = np.array([center_x, center_y])
        
        self.thermal_radius = thermal_params.get('radius', 150.0)
        # Match 'max_vertical_speed' from config to the W_z_max attribute
        self.W_z_max = thermal_params.get('max_vertical_speed', 3.0) 
        
        # Load other environment params
        self.rho = env_params.get('rho_air', 1.225)
        
    def _load_config(self, path):
        """Loads configuration from a YAML file for internal use."""
        if not os.path.exists(path):
            # Fallback path logic in case the file is run from a different directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            fallback_path = os.path.join(base_dir, '..', path) # Assumes config is one level up
            if os.path.exists(fallback_path):
                path = fallback_path
            else:
                print(f"FATAL: Configuration file not found at {path} or {fallback_path}")
                sys.exit(1)
                
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"FATAL: Could not load configuration file {path}: {e}")
            sys.exit(1)

    def get_thermal_center(self):
        """Returns the [x, y] coordinates of the thermal center."""
        return self.thermal_center

    def get_thermal_lift(self, x, y, z):
        """
        Calculates the vertical wind (Wz) at a given (x, y, z) location using
        a smooth, conical lift profile.
        """
        # Calculate horizontal distance to the thermal center
        dx = x - self.thermal_center[0]
        dy = y - self.thermal_center[1]
        horizontal_distance = np.sqrt(dx**2 + dy**2)
        
        if horizontal_distance <= self.thermal_radius:
            # Conical Profile: Lift scales linearly from W_z_max at center (dist=0) 
            # to 0 at the radius (dist=radius).
            lift_factor = 1.0 - (horizontal_distance / self.thermal_radius)
            Wz = self.W_z_max * lift_factor
            
            # Note: The altitude factor is omitted, assuming lift is constant for simplicity.
            return Wz
        else:
            return 0.0 # No lift outside the thermal radius