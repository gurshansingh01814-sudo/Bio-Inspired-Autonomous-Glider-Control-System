import numpy as np
import yaml
import sys
import os

class AtmosphericModel:
    
    def __init__(self, config_path):
        """Initializes the atmospheric model and thermal parameters."""
        
        self.config = self._load_config(config_path)
        
        # CRITICAL FIX: Access parameters based on the corrected config file structure
        atmos_params = self.config.get('ATMOSPHERE', {})
        thermal_params = atmos_params.get('THERMAL', {})
        
        self.center_x = thermal_params.get('center_x', 500.0)
        self.center_y = thermal_params.get('center_y', 500.0)
        self.radius = thermal_params.get('radius', 150.0)
        
        # CORRECTED KEY: using 'max_vertical_speed'
        self.max_lift = thermal_params.get('max_vertical_speed', 3.0) 
        
        # ASSUMPTION: Using cloud_base from config
        self.cloud_base = thermal_params.get('cloud_base', 1500.0) 
        self.ground_alt = 50.0 # Min altitude where thermal still has full strength
        
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

    def get_thermal_lift(self, x, y, z):
        """Calculates the vertical lift (Wz) at a given (x, y, z) position."""
        
        # --- 1. Radial Decay (Parabolic Model) ---
        dist_h = np.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
        
        if dist_h >= self.radius:
            radial_factor = 0.0 # Zero lift outside the defined radius
        else:
            # parabolic decay: 1 - (r/R)^2
            radial_factor = 1.0 - (dist_h / self.radius)**2 
        
        # --- 2. Altitude Decay (Realistic decay from base to cloud base) ---
        if z <= self.ground_alt or z >= self.cloud_base:
            alt_factor = 0.0
        else:
            mid_alt = (self.cloud_base + self.ground_alt) / 2.0
            
            if z < mid_alt:
                # Ramp up from ground to mid_alt
                alt_factor = (z - self.ground_alt) / (mid_alt - self.ground_alt)
            else:
                # Decay down from mid_alt to cloud_base
                alt_factor = (self.cloud_base - z) / (self.cloud_base - mid_alt)
        
        alt_factor = np.clip(alt_factor, 0.0, 1.0)
        
        # --- 3. Total Vertical Speed ---
        Wz = self.max_lift * radial_factor * alt_factor
        
        return Wz