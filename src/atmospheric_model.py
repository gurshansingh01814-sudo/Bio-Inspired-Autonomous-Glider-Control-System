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

    def get_thermal_center(self):
        """Returns the [x, y] coordinates of the thermal center."""
        return self.thermal_center

    def get_thermal_lift(self, x, y, z):
        """
        Calculates the vertical wind (Wz) at a given (x, y, z) location using
        a smooth, conical lift profile. Lift is strongest at the center and 
        drops linearly to zero at the edge.
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
            
            # Altitude Factor (Optional but helpful for realism)
            # You can add logic here if lift fades with altitude, e.g., lift * (1 - z/z_max)
            
            return Wz
        else:
            return 0.0 # No lift outside the thermal radius