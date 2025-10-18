import numpy as np
import time
import os
import pandas as pd
import sys
import yaml
import math
from datetime import datetime # Added for precise timestamping

# Assume the necessary classes are now correctly imported from src/
try:
    from src.glider_dynamics import GliderDynamics 
    from src.mpc_controller import MPCController
    from src.atmospheric_model import AtmosphericModel
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class GliderControlSystem:
    
    def __init__(self, config_path='./data/glider_config.yaml'):
        
        # Load config to retrieve simulation parameters first
        self.config_path = config_path
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_path}")
            sys.exit(1)
        
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
        self.flight_data = [] # List to hold dictionaries of data points
        self.step_count = 0
        self.current_time = 0.0
        
        # Placeholder for previous control inputs: [CL_command, phi_command]
        self.last_u_star = np.array([0.2, 0.0]) 
        
        print("System initialized successfully.")
    
    # --- NEW METHOD: Records all relevant data for CSV logging ---
    def record_data(self, X_state, U_control, Wz_lift):
        """Calculates all logged variables and appends the data dictionary to self.flight_data."""
        x, y, z, vx, vy, vz = X_state
        CL_cmd, phi_cmd = U_control
        
        # Calculate derived values (Airspeed, Distance, etc.)
        airspeed = math.sqrt(vx**2 + vy**2 + vz**2)
        thermal_center = self.thermal.get_thermal_center()
        distance = math.sqrt((x - thermal_center[0])**2 + (y - thermal_center[1])**2) 
        
        # Create the data dictionary for one time step
        data_point = {
            'time': self.current_time,
            'x': x,
            'y': y,
            'z': z,
            'vx': vx,
            'vy': vy,
            'vz': vz,
            'airspeed': airspeed, 
            'Wz': Wz_lift,
            # Use actual CL and Phi from GliderDynamics for plotting actual flight path
            'CL': self.glider.CL,       
            'Phi': self.glider.phi,     
            'CL_cmd': CL_cmd,
            'phi_cmd': phi_cmd,
            'dist_to_center': distance  
        }
        
        self.flight_data.append(data_point) # *** THE FIX: ADDING DATA TO THE LIST ***

    # --- MODIFIED METHOD: Console print only ---
    def log_status(self, X_state, U_control, Wz_lift):
        """Prints a snapshot of the simulation state to the console (for viewing only)."""
        x, y, z, vx, vy, vz = X_state
        CL, phi = U_control
        
        # Calculate Airspeed and Distance to Thermal Center
        airspeed = math.sqrt(vx**2 + vy**2 + vz**2)
        thermal_center = self.thermal.get_thermal_center()
        distance = math.sqrt((x - thermal_center[0])**2 + (y - thermal_center[1])**2)
        
        # Use the standard print function for console output (matching previous output format)
        print(f"T={self.current_time:.1f}s | Alt={z:.2f}m | Airspeed={airspeed:.2f}m/s | Wz={Wz_lift:.2f}m/s | Dist={distance:.1f}m | CL={CL:.2f} | Phi={math.degrees(phi):.1f}°")

    def main_loop(self):
        """Runs the main simulation loop until maximum time or minimum altitude is reached."""
        print("Starting simulation loop...")
        
        # Log initial state (T=0.0s) before the main loop starts
        thermal_center = self.thermal.get_thermal_center()
        X_initial = self.glider.get_state()
        Wz_initial = self.thermal.get_thermal_lift(X_initial[0], X_initial[1], X_initial[2])
        self.record_data(X_initial, self.last_u_star, Wz_initial) 
        self.log_status(X_initial, self.last_u_star, Wz_initial)
        
        # Main simulation loop
        while self.current_time < self.T_end:
            
            # --- MPC Control Step (Executed every self.dt_mpc seconds) ---
            if self.step_count % self.mpc_update_freq == 0:
                
                X_current = self.glider.get_state()
                thermal_center = self.thermal.get_thermal_center()
                
                # Predict atmospheric lift (Wz)
                thermal_center_Wz = self.thermal.get_thermal_lift(thermal_center[0], thermal_center[1], X_current[2])
                Wz_prediction = np.full((1, self.N_horizon), thermal_center_Wz)
                
                # Solve MPC problem
                u_optimal = self.mpc.solve_mpc(X_current, thermal_center, Wz_prediction)
                self.last_u_star = u_optimal
                
                # Log status every MPC step (every self.dt_mpc seconds)
                # NOTE: We log_status for console print, but record_data happens after the step
                # The console print here will use the state BEFORE the step and the command for the next step.
                
            # --- Glider Dynamics Step (Executed every self.dt_sim seconds) ---
            CL_cmd, phi_cmd = self.last_u_star[0], self.last_u_star[1]
            X_new = self.glider.update(CL_cmd, phi_cmd, self.dt_sim, self.thermal)
            
            # --- Update Time and Counters ---
            self.current_time += self.dt_sim
            self.step_count += 1
            
            # --- Record Data After State Update (CRITICAL CHANGE) ---
            # Re-calculate Wz at the NEW position for accurate logging
            Wz_actual = self.thermal.get_thermal_lift(X_new[0], X_new[1], X_new[2])
            self.record_data(X_new, self.last_u_star, Wz_actual) 
            
            # --- Check Termination Conditions ---
            Z_alt = X_new[2]
            if Z_alt <= 20.0:
                print(f"\n--- LANDING / CRASH: Minimum altitude reached at T={self.current_time:.1f}s. Simulation terminated. ---")
                break
                
            # Print console status only when an MPC step would have occurred for clean output
            if self.step_count % self.mpc_update_freq == 0:
                 self.log_status(X_new, self.last_u_star, Wz_actual)
            
        # --- Final Logging and Save ---
        self.save_results()

    def save_results(self):
        """Saves the recorded flight data to a CSV file."""
        # Use pandas to convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(self.flight_data)
        
        # Ensure the path to the 'results' folder is correct relative to the script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Use datetime for a robust timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flight_data_{timestamp}.csv"
        filepath = os.path.join(results_dir, filename)
        
        # Use to_csv to write the DataFrame contents
        if not df.empty:
            df.to_csv(filepath, index=False)
            print(f"\nSimulation results successfully saved to: {filepath}")
        else:
            print("\nWarning: No data recorded. CSV file not created.")

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
        system.main_loop() 
        # Note: save_results is called from inside main_loop, so no need to call it here.
    except Exception as e:
        print(f"\n--- FATAL SIMULATION ERROR ---")
        print(f"An unrecoverable error occurred: {e}")
        sys.exit(1)