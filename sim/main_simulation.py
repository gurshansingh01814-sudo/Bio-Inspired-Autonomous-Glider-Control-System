import numpy as np
import math
import sys
import os
import yaml # Added for robust config checking

# --- CRITICAL FIX IMPORTS ---
# Corrected package imports
# NOTE: Assumes __init__.py exists in 'src' folder
# NOTE: The user must ensure these modules (mpc_controller, glider_dynamics, atmospheric_model)
#       are available in the python environment.
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
        positive_Wz = [w for w in self.log['Wz'] if w > 0]
        avg_Wz = np.mean(positive_Wz) if positive_Wz else 0.0
        
        print("\n" + "="*50)
        print("     GLIDER CONTROL SYSTEM SIMULATION COMPLETE")
        print("="*50)
        print(f"Total Flight Time: {final_time:.1f} seconds")
        print(f"Final Altitude (z): {final_alt:.2f} meters")
        print(f"Final Position (x, y): ({self.log['x'][-1]:.1f}, {self.log['y'][-1]:.1f}) meters")
        print(f"Average Thermal Uplift (in thermal area): {avg_Wz:.2f} m/s")
        print("="*50 + "\n")

# Function to find the absolute path of the project root
def find_project_root():
    """Locates the project root directory."""
    # Assumes 'main_simulation.py' is in a subdirectory (e.g., 'sim' or 'scripts')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Going up one level to the project root
    return os.path.abspath(os.path.join(current_dir, os.pardir))


class GliderControlSystem:
    def __init__(self, config_file_name='glider_config.yaml'):
        # 1. Find Project Root and Config Path
        self.PROJECT_ROOT = find_project_root()
        # NOTE: Config file path updated to be robust, assuming 'glider_config.yaml' is at project root.
        # Based on your prompt, it seems the config file is at the root or an easily accessible path.
        self.CONFIG_PATH = os.path.join(self.PROJECT_ROOT, config_file_name) 

        print(f"Project Root: {self.PROJECT_ROOT}")
        print(f"Attempting to load config from: {self.CONFIG_PATH}")
        sys.stdout.flush() 

        # 2. Check for configuration file existence (Bulletproof Check)
        if not os.path.exists(self.CONFIG_PATH):
            print("\nFATAL ERROR: Configuration file not found at expected path.")
            print(f"Path checked: {self.CONFIG_PATH}")
            print("Please ensure 'glider_config.yaml' is in the project root or adjust the path.")
            sys.exit(1)

        # 3. Component Instantiation
        self.dynamics = GliderDynamics(self.CONFIG_PATH)
        self.atmosphere = AtmosphericModel(self.CONFIG_PATH)
        self.mpc = MPCController(self.CONFIG_PATH) # MPC now loads configuration properly

        # 4. Simulation Parameters (load from MPC's config for consistency)
        # CRITICAL FIX: Use the MPC's prediction DT (DT_PREDICT) as the simulation step.
        config_data_mpc = self.mpc.config.get('MPC', {})
        config_data_sim = self.mpc.config.get('SIMULATION', {})
        
        self.MAX_TIME = config_data_sim.get('MAX_TIME', 120.0)
        
        # Use MPC's PREDICT_DT (2.0s as per config) as the simulator step time.
        # This synchronizes the control updates with the prediction horizon.
        self.DT = config_data_mpc.get('PREDICT_DT', 2.0) 
        
        # Calculate total steps based on the synchronized DT
        self.TOTAL_STEPS = int(self.MAX_TIME / self.DT)
        
        print(f"Simulation DT set to MPC DT: {self.DT:.1f}s")
        print(f"Total steps: {self.TOTAL_STEPS}")

        # 5. Initial State: [x, y, z, vx, vy, vz]
        # Safer start conditions, well above the MPC's MIN_ALTITUDE (20.0m).
        self.current_state = np.array([0.0, 0.0, 200.0, 15.0, 0.0, 0.0]) 

        self.logger = DataLogger()

    def run(self):
        """Main simulation loop."""
        print(f"Starting Glider Simulation for {self.MAX_TIME:.1f} seconds (DT={self.DT:.1f})...")
        sys.stdout.flush()
        
        # NOTE: Thermal data is read from the atmospheric model
        thermal_center = self.atmosphere.get_thermal_center()
        thermal_radius = self.atmosphere.get_thermal_radius()
        
        print(f"Thermal Target at: ({thermal_center[0]}, {thermal_center[1]}) with Radius: {thermal_radius}m")
        
        # Initial control guess: [phi (bank), alpha (pitch)]
        current_control = np.array([0.0, 0.0])
        
        for k in range(self.TOTAL_STEPS):
            t = k * self.DT
            position = self.current_state[0:3]

            # --- 1. Environmental Winds ---
            W_atm = self.atmosphere.get_wind_vector(position)
            W_atm_z = W_atm[2]

            # --- 2. Compute Control (MPC) ---
            # NOTE: MPC expects thermal_center and thermal_radius separately for parameter vector P
            current_control = self.mpc.compute_control(
                self.current_state, 
                thermal_center, 
                thermal_radius
            )
            
            # --- 3. Step Dynamics ---
            # CRITICAL: Pass the synchronized simulation DT (self.DT) to the dynamics model
            next_state = self.dynamics.step(
                self.current_state, 
                current_control[0], 
                current_control[1], 
                W_atm, 
                self.DT
            )
            self.current_state = next_state
            
            # --- 4. Log Data ---
            dist_to_thermal = np.linalg.norm(position[0:2] - thermal_center)
            self.logger.log_step(t + self.DT, self.current_state, current_control, W_atm_z, dist_to_thermal)
            
            # Termination Condition
            if self.current_state[2] < 5.0 and k > 0:
                print(f"\nAltitude too low. Terminating simulation at t={t+self.DT:.1f}s.")
                sys.stdout.flush()
                break
                
            # Feedback print every 10 steps
            if k % 10 == 0:
                # Calculate approximate air speed for logging
                V_air_vec = self.current_state[3:6] - W_atm
                V_air_approx = np.linalg.norm(V_air_vec)
                
                print(f"T={t+self.DT:.1f}s | Alt={self.current_state[2]:.2f}m | Airspeed={V_air_approx:.2f}m/s | Wz={W_atm_z:.2f}m/s | Dist={dist_to_thermal:.1f}m")
                sys.stdout.flush() 
        
        # ------------------ Results ------------------
        self.logger.print_summary()


if __name__ == '__main__':
    # Add project root to path for module execution safety
    try:
        project_root = find_project_root()
        if project_root not in sys.path:
            sys.path.append(project_root)
        
        # Add src folder to path for internal imports
        src_path = os.path.join(project_root, 'src')
        if src_path not in sys.path:
            sys.path.append(src_path)
            
        system = GliderControlSystem()
        system.run()
        
    except Exception as e:
        print(f"\n--- UNEXPECTED SIMULATION FAILURE ---")
        print(f"A component failed to run after initialization: {e}")
        # Print the traceback for full debugging information
        import traceback
        traceback.print_exc()
        sys.exit(1)