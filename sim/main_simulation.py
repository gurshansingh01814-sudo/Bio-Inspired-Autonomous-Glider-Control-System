import numpy as np
import time
import os
import pandas as pd
import sys
import yaml
import math # Needed for math.degrees, math.radians etc.

# Assume the necessary classes are now correctly imported from src/
try:
    from src.glider_model import GliderDynamics 
    from src.mpc_controller import MPCController
    from src.atmospheric_model import AtmosphericModel
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class GliderControlSystem:
    
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
        self.current_time = 0.0
        
        # Placeholder for previous control inputs: [CL_command, phi_command]
        # FIX: Ensure initial control is a legal, minimum CL glide. (CL_MIN is 0.2)
        self.last_u_star = np.array([0.2, 0.0]) 
        
        print("System initialized successfully.")
    
    # --- MISSING main_loop METHOD ADDED HERE ---
    def main_loop(self):
        """Runs the main simulation loop until maximum time or minimum altitude is reached."""
        print("Starting simulation loop...")
        
        # Main simulation loop
        while self.current_time < self.T_end:
            
            # --- MPC Control Step (Executed every self.dt_mpc seconds) ---
            if self.step_count % self.mpc_update_freq == 0:
                
                # 1. Get current state and thermal data
                X_current = self.glider.get_state()
                thermal_center = self.thermal.get_thermal_center()
                
                # 2. Predict atmospheric lift (Wz) over the horizon
                # Simplification: Assume Wz is constant (thermal_center_Wz) over the horizon
                thermal_center_Wz = self.thermal.get_thermal_lift(thermal_center[0], thermal_center[1], X_current[2])
                Wz_prediction = np.full((1, self.N_horizon), thermal_center_Wz)
                
                # 3. Solve MPC problem
                u_optimal = self.mpc.solve_mpc(X_current, thermal_center, Wz_prediction)
                self.last_u_star = u_optimal
                
                # Log status every MPC step (every self.dt_mpc seconds)
                self.log_status(X_current, self.last_u_star, thermal_center_Wz)

            # --- Glider Dynamics Step (Executed every self.dt_sim seconds) ---
            CL_cmd, phi_cmd = self.last_u_star[0], self.last_u_star[1]
            X_new = self.glider.update(CL_cmd, phi_cmd, self.dt_sim, self.thermal)
            
            # --- Check Termination Conditions ---
            Z_alt = X_new[2]
            if Z_alt <= 20.0:
                print(f"\n--- LANDING / CRASH: Minimum altitude reached at T={self.current_time:.1f}s. Simulation terminated. ---")
                break
                
            # --- Update Time and Counters ---
            self.current_time += self.dt_sim
            self.step_count += 1
            
        # --- Final Logging and Save ---
        self.save_results()

    def log_status(self, X_state, U_control, Wz_lift):
        """Prints a snapshot of the simulation state to the console."""
        x, y, z, vx, vy, vz = X_state
        CL, phi = U_control
        
        # Calculate Airspeed and Distance to Thermal Center
        airspeed = math.sqrt(vx**2 + vy**2 + vz**2)
        thermal_center = self.thermal.get_thermal_center()
        distance = math.sqrt((x - thermal_center[0])**2 + (y - thermal_center[1])**2)
        
        print(f"T={self.current_time:.1f}s | Alt={z:.2f}m | Airspeed={airspeed:.2f}m/s | Wz={Wz_lift:.2f}m/s | Dist={distance:.1f}m | CL={CL:.2f} | Phi={math.degrees(phi):.1f}°")

    def save_results(self):
        """Saves the recorded flight data to a CSV file."""
        df = pd.DataFrame(self.flight_data)
        
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"flight_data_{timestamp}.csv"
        filepath = os.path.join(results_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"\nSimulation results saved to: {filepath}")

# --- Main execution block ---
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'data', 'glider_config.yaml')
    
    if not os.path.exists(config_path):
        print(f"FATAL: Configuration file not found at {config_path}")
        sys.exit(1)
        
    try:
        system = GliderControlSystem(config_path)
        system.main_loop() # This is the line that caused the error
    except Exception as e:
        print(f"\n--- FATAL SIMULATION ERROR ---")
        print(f"An unrecoverable error occurred: {e}")
        sys.exit(1)