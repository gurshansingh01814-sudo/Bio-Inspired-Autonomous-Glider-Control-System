import numpy as np
import time
import os
import pandas as pd
import sys
import yaml

# Assume the necessary classes are now correctly imported from src/
try:
    from src.glider_model import GliderDynamics 
    from src.mpc_controller import MPCController
    from src.atmospheric_model import AtmosphericModel
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class GliderControlSystem:
    # ... (rest of __init__ code) ...

    def __init__(self, config_path='./data/glider_config.yaml'):
        
        # Load config to retrieve simulation parameters first
        self.config_path = config_path
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Simulation parameters
        sim_params = config.get('SIMULATION', {})
        self.T_end = sim_params.get('max_time', 120.0)      
        self.dt_sim = sim_params.get('time_step_s', 0.2)    
        
        mpc_params = config.get('MPC', {})
        self.dt_mpc = mpc_params.get('control_rate_s', 2.0) 
        self.N_horizon = mpc_params.get('horizon_steps', 40)
        
        # CRITICAL FIX: Use integer steps for reliable timing
        self.mpc_update_freq = int(self.dt_mpc / self.dt_sim)
        if self.mpc_update_freq < 1:
            raise ValueError("dt_mpc must be an integer multiple of dt_sim!")

        # Initialize sub-systems
        self.glider = GliderDynamics(config_path)
        self.thermal = AtmosphericModel(config_path)
        self.mpc = MPCController(config_path)
        
        # Initialize storage and counters
        self.flight_data = []
        self.step_count = 0
        
        # Placeholder for previous control inputs: [CL_command, phi_command]
        # FIX: Ensure initial control is a legal, minimum CL glide. (CL_MIN is 0.2)
        self.last_u_star = np.array([0.2, 0.0]) 
        
        print("System initialized successfully.")
    
    # ... (rest of the class methods) ...

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'data', 'glider_config.yaml')
    
    # --- CRITICAL FIX: Use os.path.exists() instead of config_path.exists() ---
    if not os.path.exists(config_path):
        print(f"FATAL: Configuration file not found at {config_path}")
        sys.exit(1)
    # --- END FIX ---
    
    try:
        system = GliderControlSystem(config_path)
        system.main_loop()
    except Exception as e:
        print(f"\n--- FATAL SIMULATION ERROR ---")
        print(f"An unrecoverable error occurred: {e}")
        sys.exit(1)