import numpy as np
import yaml
import sys
import os

class AtmosphericModel:
    
    def __init__(self, config_path):
        # Load configuration using the provided path
        self.config = self._load_config(config_path)
        
        # Thermal parameters are typically nested under 'THERMAL' in the config
        thermal_params = self.config.get('THERMAL', {})
        
        # CRITICAL FIX: Define the required attributes for main_simulation.py
        self.center_x = thermal_params.get('center_x', 500.0)
        self.center_y = thermal_params.get('center_y', 500.0)
        
        # Other necessary parameters for thermal dynamics
        self.radius = thermal_params.get('radius', 100.0)
        self.max_lift = thermal_params.get('max_lift', 4.0)
        self.decay_alt = thermal_params.get('decay_altitude', 1500.0)
        
        print(f"Atmospheric Model (Thermal) initialized at ({self.center_x:.0f}, {self.center_y:.0f}) m.")


    def _load_config(self, path):
        """Loads configuration from a YAML file for internal use."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"FATAL: Could not load configuration file {path} in AtmosphericModel: {e}")
            sys.exit(1)

    def get_thermal_lift(self, x, y, z):
        """Calculates the vertical lift (Wz) at a given (x, y, z) position."""
        
        # Horizontal distance from thermal center
        dist_h = np.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
        
        # 1. Radial decay (Gaussian-like)
        radial_decay = np.exp(-0.5 * (dist_h / (self.radius / 2.0))**2)
        
        # 2. Altitude decay 
        if z > self.decay_alt or z < 20.0:
            alt_decay = 0.0
        else:
            # Linear decay from max lift at lower altitude to 0 at decay_alt
            alt_decay = np.clip(1.0 - (z / self.decay_alt), 0.0, 1.0)
        
        Wz = self.max_lift * radial_decay * alt_decay
        
        return Wz