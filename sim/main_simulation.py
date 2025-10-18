import numpy as np
import time
import os
import pandas as pd
import sys
import yaml # Added for robust config loading

# Assume the necessary classes are now correctly imported from src/
try:
    # Renaming for clarity in the main file
    from src.glider_dynamics import GliderDynamics 
    from src.mpc_controller import MPCController
    from src.atmospheric_model import AtmosphericModel
    from src.atmospheric_model import deg_to_rad # Helper for logging phi
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class GliderControlSystem:
    """
    Manages the complete simulation, integrating the glider dynamics, 
    the atmospheric model, and the MPC controller.
    """

    def __init__(self, config_path='./data/glider_config.yaml'):
        
        # Load config to retrieve simulation parameters first
        self.config_path = config_path
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Simulation parameters (CRITICAL FIX: Load from config)
        sim_params = config.get('SIMULATION', {})
        self.T_end = sim_params.get('max_time', 120.0)      # Total simulation time
        self.dt_sim = sim_params.get('time_step_s', 0.2)    # Simulation time step (RK4)
        
        mpc_params = config.get('MPC', {})
        self.dt_mpc = mpc_params.get('control_rate_s', 2.0) # MPC control frequency
        self.N_horizon = mpc_params.get('horizon_steps', 40)
        
        # CRITICAL FIX: Use integer steps for reliable timing
        self.mpc_update_freq = int(self.dt_mpc / self.dt_sim)
        if self.mpc_update_freq < 1:
            raise ValueError("dt_mpc must be an integer multiple of dt_sim!")

        # Initialize sub-systems (now with correct time parameters loaded)
        self.glider = GliderDynamics(config_path)
        self.thermal = AtmosphericModel(config_path)
        self.mpc = MPCController(config_path)
        
        # Initialize storage and counters
        self.flight_data = []
        self.step_count = 0
        
        # Placeholder for previous control inputs (MUST BE CL, PHI)
        # Default to a moderate glide: CL=0.8, Phi=0.0
        self.last_u_star = np.array([0.8, 0.0]) 
        
        print("System initialized successfully.")
        print(f"Simulation: {self.dt_sim}s | Control: {self.dt_mpc}s (1 update per {self.mpc_update_freq} steps)")

    def _predict_wz(self, X, U_sequence, thermal_model):
        """
        Predicts Wz over the MPC horizon. 
        Simplified: Re-evaluate thermal Wz at the current location N times.
        ADVANCED: Integrate dynamics over the predicted U_sequence to find future Wz.
        """
        # For a static thermal, Wz at the start of the horizon is good enough for a first run
        # Correct implementation requires predicting the path X(k+1) given U(k)
        
        Wz_prediction = np.zeros((1, self.N_horizon))
        x, y, z = X[0], X[1], X[2]
        
        # We will use the Wz at the *current* location for the entire horizon (Simplification)
        current_wz = thermal_model.get_thermal_lift(x, y, z)
        Wz_prediction[0, :] = current_wz 
        
        return Wz_prediction
        

    def main_loop(self):
        """Runs the main time-stepping simulation."""
        
        print("Starting simulation loop...")
        
        current_state = self.glider.get_state()
        t = 0.0
        
        while t < self.T_end:
            
            x, y, z, vx, vy, vz = current_state
            
            # Crash Check (Using Z_MIN from config file is safer, but 20.0 is fine)
            if z <= 20.0: 
                print(f"\n--- LANDING / CRASH: Minimum altitude reached at T={t:.1f}s. Simulation terminated. ---")
                break
                
            # --- MPC Control Update (Run every self.mpc_update_freq steps) ---
            if self.step_count % self.mpc_update_freq == 0:
                
                thermal_center = np.array([self.thermal.center_x, self.thermal.center_y])
                
                # CRITICAL FIX: The current state is used for the prediction
                Wz_prediction = self._predict_wz(current_state, None, self.thermal) 

                u_star = self.mpc.solve_mpc(current_state, thermal_center, Wz_prediction) 
                
                # Update last_u_star with the new command if solver converged
                # u_star is [CL_cmd, phi_cmd]
                if not np.allclose(u_star, np.array([0.8, 0.0])): # Check if it returned the safe glide command
                    self.last_u_star = u_star

            # Get the current command: last_u_star = [CL_command, phi_command]
            CL_command = self.last_u_star[0]
            phi_command = self.last_u_star[1]

            # --- Dynamics Integration (Run every dt_sim seconds) ---
            # CRITICAL FIX: Correctly pass CL_command and phi_command to the dynamics model
            new_state = self.glider.update(CL_command, phi_command, self.dt_sim, self.thermal)
            current_state = new_state
            t += self.dt_sim
            self.step_count += 1
            
            # Calculate metrics
            airspeed = np.linalg.norm(current_state[3:6])
            w_z = self.thermal.get_thermal_lift(current_state[0], current_state[1], current_state[2])
            dist_to_thermal = np.linalg.norm(current_state[0:2] - np.array([self.thermal.center_x, self.thermal.center_y]))
            
            # Log data
            self.flight_data.append([
                t, current_state[0], current_state[1], current_state[2], 
                airspeed, w_z, dist_to_thermal, CL_command, phi_command # CRITICAL FIX: Logging CL and Phi
            ])

            # Print status every 2.0 seconds (for user feedback)
            if self.step_count % self.mpc_update_freq == 0:
                print(f"T={t:.1f}s | Alt={z:.2f}m | Airspeed={airspeed:.2f}m/s | Wz={w_z:.2f}m/s | Dist={dist_to_thermal:.1f}m | CL={CL_command:.2f} | Phi={np.degrees(phi_command):.1f}°")

        self._save_results()

    def _save_results(self):
        """Saves the simulation data to a CSV file."""
        df = pd.DataFrame(self.flight_data, columns=[
            'time', 'x', 'y', 'z', 'Airspeed', 'Wz', 'dist_to_center', 'CL_cmd', 'phi_cmd' # CRITICAL FIX: Column names match plotter
        ])
        
        output_dir = './results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"flight_data_{timestamp}.csv")
        df.to_csv(filepath, index=False)
        
        print(f"\nSimulation results saved to: {filepath}")
        print("Simulation successful. Run 'python scripts/plot_results.py' to visualize the flight data.")


if __name__ == '__main__':
    # Determine the configuration path relative to the current working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'data', 'glider_config.yaml')
    
    try:
        # Initialize and run the system
        system = GliderControlSystem(config_path)
        system.main_loop()
    except Exception as e:
        print(f"\n--- FATAL SIMULATION ERROR ---")
        print(f"An unrecoverable error occurred: {e}")
        sys.exit(1)
