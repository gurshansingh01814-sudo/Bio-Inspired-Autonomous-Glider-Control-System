import numpy as np
import yaml
import time
import os
import sys
import pandas as pd
# Add these imports at the top
import datetime
# Other imports like GliderDynamics, MPCController, Thermal, etc. are assumed to be present

# ... (The GliderControlSystem class definition remains mostly the same, 
# but we add a 'results' attribute and modify the main_loop method) ...

class GliderControlSystem:
    # ... (rest of __init__ and other methods) ...

    def main_loop(self):
        # Initial conditions and simulation setup
        T_start = 0.0
        T_end = 400.0
        DT = self.mpc.DT # 2.0s
        current_state = self.glider.get_state()
        
        # New: Initialize data logging
        self.results = []
        
        print("Starting simulation loop...")

        t = T_start
        
        while t < T_end:
            
            # --- MPC Step ---
            thermal_center = np.array([self.thermal.center_x, self.thermal.center_y])
            thermal_radius = self.thermal.radius
            
            # Compute optimal control (bank angle phi, pitch alpha)
            u_star = self.mpc.compute_control(current_state, thermal_center, thermal_radius)
            phi_command, alpha_command = u_star[0], u_star[1]
            
            # --- Logging Data ---
            self.results.append({
                'time': t,
                'x': current_state[0],
                'y': current_state[1],
                'z': current_state[2],
                'vx': current_state[3],
                'vy': current_state[4],
                'vz': current_state[5],
                'phi': phi_command,
                'alpha': alpha_command,
                'Wz': self.thermal.get_thermal_lift(current_state[0], current_state[1], current_state[2]),
                'dist_to_center': np.sqrt((current_state[0] - self.thermal.center_x)**2 + (current_state[1] - self.thermal.center_y)**2)
            })

            # --- Dynamics Step ---
            # Glider updates its state over the time interval DT
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
            
            # Simple 500 meter distance exit condition (optional)
            if self.results[-1]['dist_to_center'] > 1000:
                print("\n--- Too far from thermal. Simulation terminated. ---")
                break

        self.save_results()


    def save_results(self):
        """Saves the recorded flight data to a timestamped CSV file."""
        if not hasattr(self, 'results') or not self.results:
            return

        # Create a results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate a timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(results_dir, f"flight_data_{timestamp}.csv")
        
        # Convert list of dictionaries to a pandas DataFrame and save
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        print(f"\nSimulation results saved to: {filepath}")

        # Store the path for the plotting script
        self.last_results_path = filepath


if __name__ == '__main__':
    # Ensure pandas is available
    try:
        import pandas as pd
    except ImportError:
        print("Pandas is required for data logging and saving. Please install it: pip install pandas")
        sys.exit(1)
        
    # Ensure matplotlib is available for the next script
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is required for plotting. Please install it: pip install matplotlib")
        # Do not exit here, allow the simulation to run and save data
        
    system = GliderControlSystem()
    system.main_loop()

    # Pass the path to the plotter if available
    if hasattr(system, 'last_results_path') and 'matplotlib.pyplot' in sys.modules:
        plot_script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'plot_results.py')
        print(f"\nPlotting script will try to load data from: {system.last_results_path}")
        # Note: In a real system, you would execute the plotting script here.
        # For this environment, we just instruct the user to run the next script.