import numpy as np
import yaml
import time
import os
import sys
import datetime
import pandas as pd

# CRITICAL: Ensure these imports point to the correct files.
try:
    from src.mpc_controller import MPCController
    # Assuming GliderDynamics and Thermal classes are defined or imported from files
    from sim.glider_dynamics import GliderDynamics 
    from sim.thermal_model import Thermal 
except ImportError as e:
    print(f"FATAL: Missing a required class import: {e}")
    # If the above fails, you might need to adjust your system's Python path or the import style
    print("If imports fail, ensure you are running the script using: python -m sim.main_simulation")
    sys.exit(1)


class GliderControlSystem:
    # --- Configuration Paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # FIX: Corrected path from 'config' to 'data'
    CONFIG_PATH = os.path.join(BASE_DIR, 'data', 'glider_config.yaml') 

    def __init__(self):
        print(f"Attempting to load config from: {self.CONFIG_PATH}")
        self.config = self._load_config(self.CONFIG_PATH)

        # --- Initialization of Components (CRITICAL FOR SELF.MPC) ---
        self.glider = GliderDynamics(self.config)
        self.thermal = Thermal(self.config)
        
        # This line must execute successfully to create self.mpc
        self.mpc = MPCController(self.CONFIG_PATH) 
        
        # Data logging initialization
        self.results = []
        self.last_results_path = None
        
        print("System initialized successfully.")

    def _load_config(self, path):
        """Loads configuration from a YAML file."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            # FATAL: Configuration loading failure
            print(f"FATAL: Could not load configuration file {path}: {e}")
            sys.exit(1)

    # --- MAIN LOOP ---
    def main_loop(self):
        T_end = 400.0
        DT = self.mpc.DT 
        
        current_state = self.glider.get_state()
        print("Starting simulation loop...")

        t = 0.0
        
        while t < T_end:
            
            # --- MPC Step ---
            thermal_center = np.array([self.thermal.center_x, self.thermal.center_y])
            thermal_radius = self.thermal.radius
            
            u_star = self.mpc.compute_control(current_state, thermal_center, thermal_radius)
            phi_command, alpha_command = u_star[0], u_star[1]
            
            # --- Logging Data ---
            current_state_list = current_state.tolist()
            self.results.append({
                'time': t,
                'x': current_state_list[0],
                'y': current_state_list[1],
                'z': current_state_list[2],
                'vx': current_state_list[3],
                'vy': current_state_list[4],
                'vz': current_state_list[5],
                'phi': phi_command,
                'alpha': alpha_command,
                'Wz': self.thermal.get_thermal_lift(current_state[0], current_state[1], current_state[2]),
                'dist_to_center': np.sqrt((current_state[0] - self.thermal.center_x)**2 + (current_state[1] - self.thermal.center_y)**2)
            })

            # --- Dynamics Step ---
            next_state = self.glider.update(phi_command, alpha_command, DT, self.thermal)
            current_state = next_state
            t += DT

            # Print status update
            airspeed = np.linalg.norm(current_state[3:])
            print(f"T={t:.1f}s | Alt={current_state[2]:.2f}m | Airspeed={airspeed:.2f}m/s | Wz={self.results[-1]['Wz']:.2f}m/s | Dist={self.results[-1]['dist_to_center']:.1f}m")
            
            # Exit conditions
            if current_state[2] <= 5.0:
                print("\n--- LANDING / CRASH: Minimum altitude reached. Simulation terminated. ---")
                break
            
            if self.results[-1]['dist_to_center'] > 1000:
                print("\n--- Too far from thermal. Simulation terminated. ---")
                break

        self.save_results()


    def save_results(self):
        """Saves the recorded flight data to a timestamped CSV file."""
        if not hasattr(self, 'results') or not self.results:
            return

        results_dir = os.path.join(self.BASE_DIR, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(results_dir, f"flight_data_{timestamp}.csv")
        
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        print(f"\nSimulation results saved to: {filepath}")
        self.last_results_path = filepath


if __name__ == '__main__':
    # Ensure dependencies are available for data logging
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: Please run 'pip install pandas matplotlib' to enable data logging and plotting.")
        sys.exit(1)
        
    system = GliderControlSystem()
    system.main_loop()

    if hasattr(system, 'last_results_path') and system.last_results_path:
        # Inform the user how to plot the data
        print(f"\nSimulation successful. Run 'python scripts/plot_results.py' to visualize the flight data stored in {system.last_results_path}")
