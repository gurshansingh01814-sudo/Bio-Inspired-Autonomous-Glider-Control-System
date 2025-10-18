import numpy as np
import math
import sys
import os

# --- CRITICAL FIX IMPORTS ---
# Corrected package imports
from src.mpc_controller import MPCController
from src.glider_dynamics import GliderDynamics
from src.atmospheric_model import AtmosphericModel
# ---------------------------

# NOTE: The DataLogger class is included here for completeness
class DataLogger:
    """A simple class to log simulation data and print a summary."""
    def __init__(self):
        self.log = {
            'time': [],
            'x': [], 'y': [], 'z': [], 'vx': [], 'vy': [], 'vz': [],
            'phi': [], 'alpha': [], 
            'Wz': [], 'dist_to_thermal': []
        }

    def log_step(self, t, state, control, W_atm_z, dist_to_thermal):
        """Records the state and control at the given time step."""
        self.log['time'].append(t)
        self.log['x'].append(state[0])
        self.log['y'].append(state[1])
        self.log['z'].append(state[2])
        self.log['vx'].append(state[3])
        self.log['vy'].append(state[4])
        self.log['vz'].append(state[5])
        self.log['phi'].append(control[0])
        self.log['alpha'].append(control[1])
        self.log['Wz'].append(W_atm_z)
        self.log['dist_to_thermal'].append(dist_to_thermal)

    def print_summary(self):
        """Prints a summary of the simulation results."""
        if not self.log['time']:
            print("\n--- Simulation Summary ---\nNo data logged.")
            return

        final_time = self.log['time'][-1]
        final_alt = self.log['z'][-1]
        
        # Calculate average thermal usage
        avg_Wz = np.mean([w for w in self.log['Wz'] if w > 0]) # Only average positive uplift
        
        print("\n" + "="*50)
        print("    GLIDER CONTROL SYSTEM SIMULATION COMPLETE")
        print("="*50)
        print(f"Total Flight Time: {final_time:.1f} seconds")
        print(f"Final Altitude (z): {final_alt:.2f} meters")
        print(f"Final Position (x, y): ({self.log['x'][-1]:.1f}, {self.log['y'][-1]:.1f}) meters")
        print(f"Average Thermal Uplift: {avg_Wz:.2f} m/s")
        print("="*50 + "\n")

# Function to find the absolute path of the project root
def find_project_root():
    """Locates the project root directory (Bio-Inspired-Autonomous-Glider-Control-System)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming 'sim' is directly under the project root
    return os.path.abspath(os.path.join(current_dir, os.pardir))


class GliderControlSystem:
    def __init__(self, config_file_name='glider_config.yaml'):
        # 1. Find Project Root
        self.PROJECT_ROOT = find_project_root()
        self.CONFIG_PATH = os.path.join(self.PROJECT_ROOT, 'data', config_file_name)

        print(f"Project Root: {self.PROJECT_ROOT}")
        print(f"Attempting to load config from: {self.CONFIG_PATH}")
        sys.stdout.flush() 

        # 2. Check for configuration file existence
        if not os.path.exists(self.CONFIG_PATH):
             # CRITICAL check to prevent the 'str' error if the file isn't found
             print("FATAL ERROR: Configuration file not found at expected path.")
             print("Please ensure 'data/glider_config.yaml' exists.")
             sys.exit(1) # Exit immediately before any component tries to load it incorrectly
        
        # 3. Component Instantiation
        # Components load config using the known, absolute path
        self.dynamics = GliderDynamics(self.CONFIG_PATH)
        self.atmosphere = AtmosphericModel(self.CONFIG_PATH)
        self.mpc = MPCController(self.CONFIG_PATH)

        # 4. Simulation Parameters (loaded from MPC controller's config, which is guaranteed to be a dict)
        self.MAX_TIME = self.mpc.config.get('SIMULATION', {}).get('MAX_TIME', 120.0)
        self.DT = self.mpc.config.get('SIMULATION', {}).get('DT', 1.0)
        self.TOTAL_STEPS = int(self.MAX_TIME / self.DT)

        # Initial State: [x, y, z, vx, vy, vz]
        self.current_state = np.array([0.0, 0.0, 150.0, 15.0, 0.0, -1.0]) 

        self.logger = DataLogger()

    def run(self):
        """Main simulation loop."""
        print(f"Starting Glider Simulation for {self.MAX_TIME:.1f} seconds (DT={self.DT:.1f})...")
        sys.stdout.flush()
        
        thermal_center = self.atmosphere.get_thermal_center()
        thermal_radius = self.atmosphere.get_thermal_radius()
        
        # Initial control guess: [phi (bank), alpha (pitch)]
        current_control = np.array([0.0, 0.0])
        
        for k in range(self.TOTAL_STEPS):
            t = k * self.DT
            position = self.current_state[0:3]

            # --- 1. Environmental Winds ---
            W_atm = self.atmosphere.get_wind_vector(position)
            W_atm_z = W_atm[2]

            # --- 2. Compute Control (MPC) ---
            current_control = self.mpc.compute_control(self.current_state, thermal_center, thermal_radius)
            
            # --- 3. Step Dynamics ---
            next_state = self.dynamics.step(self.current_state, current_control[0], current_control[1], W_atm, self.DT)
            self.current_state = next_state
            
            # --- 4. Log Data ---
            dist_to_thermal = np.linalg.norm(position[0:2] - thermal_center)
            self.logger.log_step(t + self.DT, self.current_state, current_control, W_atm_z, dist_to_thermal)
            
            # Termination Condition
            if self.current_state[2] < 5.0 and k > 0:
                print(f"Altitude too low. Terminating simulation at t={t+self.DT:.1f}s.")
                sys.stdout.flush()
                break
                
            # Feedback print every 10 steps
            if k % 10 == 0:
                V_air_approx = np.linalg.norm(self.current_state[3:6] - W_atm)
                print(f"T={t+self.DT:.1f}s | Alt={self.current_state[2]:.2f}m | Airspeed={V_air_approx:.2f}m/s | Wz={W_atm_z:.2f}m/s")
                sys.stdout.flush() 
        
        # ------------------ Results ------------------
        self.logger.print_summary()


if __name__ == '__main__':
    # Add project root to path for execution from IDE/different directory
    sys.path.append(find_project_root())
    
    try:
        # Check if the sim/main_simulation.py script is being run directly from the sim folder
        # If so, we need to add the parent directory to the path so python can find the 'src' package.
        if os.path.basename(os.getcwd()) == 'sim':
            sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
            
        system = GliderControlSystem()
        system.run()
        
    except Exception as e:
        print(f"\n--- UNEXPECTED SIMULATION FAILURE ---")
        print(f"A component failed to run after initialization: {e}")
        # Print the traceback for full debugging information
        import traceback
        traceback.print_exc()
        sys.exit(1)
