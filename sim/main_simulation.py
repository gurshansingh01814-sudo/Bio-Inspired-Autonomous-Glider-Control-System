import numpy as np
import pandas as pd
import os
import time
import sys

# --- GUARANTEED PROJECT ROOT FIX ---
# Ensures Python can find the 'src' package from within the 'sim' directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------------

# --- Corrected Project Module Imports ---
# We now import from the 'src' package
try:
    from src.glider_dynamics import GliderDynamics
    from src.mpc_controller import MPCController
    from src.atmospheric_model import AtmosphericModel 
except ImportError as e:
    print(f"FATAL ERROR: Could not import required module from src/. Please ensure all files are in the src/ folder. Error: {e}")
    sys.exit(1)

# Default configuration path
DEFAULT_CONFIG_PATH = 'data/glider_config.yaml'

class GliderControlSystem:
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        
        # --- Module Initialization (Correct Call) ---
        # Only passing the config path; modules load their own parameters
        self.glider_dynamics = GliderDynamics(config_path)
        self.mpc = MPCController(config_path)
        self.atmosphere = AtmosphericModel(config_path) 
        
        # Access config from MPC (since it successfully loaded it)
        self.config = self.mpc.config 
        self.T_END = self.config.get('SIMULATION', {}).get('MAX_TIME', 120.0)
        self.DT = self.config.get('SIMULATION', {}).get('DT', 1.0)
        self.log_data = []
        
        # Ensure the data directory exists for saving logs
        data_path = os.path.join(project_root, 'data')
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    def run_simulation(self, X0):
        
        print(f"🚀 Starting Autonomous Glider Control Simulation for {self.T_END} seconds (DT={self.DT}s)...")
        print("-" * 50)
        
        current_time = 0.0
        Xk = X0.copy()

        # Log Initial State (t=0)
        initial_log_entry = {
            'time': current_time, 
            'x': Xk[0], 'y': Xk[1], 'z': Xk[2], 
            'vx': Xk[3], 'vy': Xk[4], 'vz': Xk[5],
            'phi': 0.0, 'alpha': 0.0, 
            'thermal_wz': self.atmosphere.get_wind_vector(Xk[0:3])[2],
            'thermal_detected': self.atmosphere.get_thermal_state(Xk[0:3])
        }
        self.log_data.append(initial_log_entry)
        current_time += self.DT


        # --- MAIN SIMULATION LOOP ---
        while current_time <= self.T_END:
            
            # 1. PERCEPTION
            thermal_center, thermal_radius = self.atmosphere.get_thermal_center(), self.atmosphere.get_thermal_radius()
            W_atm = self.atmosphere.get_wind_vector(Xk[0:3])

            # 2. CONTROL (Pass required thermal information)
            try:
                U_next = self.mpc.compute_control(Xk, thermal_center, thermal_radius)
            except RuntimeError as e:
                print(f"\n⚠️ MPC Solver Failed at Time {current_time:.1f}s: {e}. Defaulting to safe glide.")
                U_next = np.array([0.0, np.deg2rad(5.0)]) # Default to straight glide
            except Exception as e:
                print(f"\n❌ Unhandled MPC Exception at Time {current_time:.1f}s: {e}. Terminating.")
                break

            phi, alpha = U_next[0], U_next[1]

            # 3. DYNAMICS
            Xk_next = self.glider_dynamics.step(Xk, phi, alpha, W_atm, self.DT)
            
            # 4. LOGGING
            if int(current_time/self.DT) % 10 == 0:
                 print(f"Time: {current_time:.1f}s | Altitude: {Xk_next[2]:.1f}m | Thermal State: {self.atmosphere.get_thermal_state(Xk_next[0:3])}")
            
            log_entry = {
                'time': current_time, 
                'x': Xk_next[0], 'y': Xk_next[1], 'z': Xk_next[2], 
                'vx': Xk_next[3], 'vy': Xk_next[4], 'vz': Xk_next[5],
                'phi': np.rad2deg(phi), 'alpha': np.rad2deg(alpha),
                'thermal_wz': W_atm[2],
                'thermal_detected': self.atmosphere.get_thermal_state(Xk_next[0:3])
            }
            self.log_data.append(log_entry)
            
            # 5. Update State and Time
            Xk = Xk_next
            current_time += self.DT

            # Safety Check: Stop if glider hits the ground
            if Xk[2] <= 5.0:
                print("\n⚠️ Simulation terminated: Glider crashed (Altitude < 5m).")
                break

        # --- END OF SIMULATION ---
        print("-" * 50)
        
        save_path = os.path.join(project_root, 'data', 'flight_log.csv')
        
        log_df = pd.DataFrame(self.log_data)
        log_df.to_csv(save_path, index=False) 

        print(f"✅ Simulation complete. Logged {len(self.log_data)} rows of data.")
        print(f"✅ Data saved successfully to: {save_path}") 

        return log_df

if __name__ == '__main__':
    # Initial State Vector X0: [x, y, z, vx, vy, vz] (Position and Velocity)
    X_initial = np.array([450.0, 450.0, 150.0, 15.0, 0.0, -1.0]) 

    # Note: MPCController will load config (including initial thermal parameters)
    try:
        controller = GliderControlSystem()
        controller.run_simulation(X_initial)
    except FileNotFoundError:
        print("\nERROR: Configuration file not found. Ensure 'data/glider_config.yaml' exists.")
    except Exception as e:
        print(f"\nUNEXPECTED SIMULATION ERROR: {e}")