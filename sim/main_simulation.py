import numpy as np
import time
import os
import pandas as pd
import sys
import yaml

# Assume the necessary classes are now correctly imported from src/
try:
    from src.glider_dynamics import GliderDynamics 
    from src.mpc_controller import MPCController
    from src.atmospheric_model import AtmosphericModel
    # CRITICAL FIX: Removed incorrect and redundant import of deg_to_rad
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
        self.last_u_star = np.array([0.2, 0.0]) 
        
        print("System initialized successfully.")

    def _predict_wz(self, X, U_sequence, thermal_model):
        """
        Predicts Wz over the MPC horizon. Simplified to current Wz for all steps.
        """
        Wz_prediction = np.zeros((1, self.N_horizon))
        x, y, z = X[0], X[1], X[2]
        
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
            
            if z <= 20.0: 
                print(f"\n--- LANDING / CRASH: Minimum altitude reached at T={t:.1f}s. Simulation terminated. ---")
                break
                
            # --- MPC Control Update ---
            if self.step_count % self.mpc_update_freq == 0:
                
                thermal_center = np.array([self.thermal.center_x, self.thermal.center_y])
                Wz_prediction = self._predict_wz(current_state, None, self.thermal) 

                u_star = self.mpc.solve_mpc(current_state, thermal_center, Wz_prediction) 
                
                if not np.allclose(u_star, np.array([0.8, 0.0])): 
                    self.last_u_star = u_star

            # Get the current command: [CL_command, phi_command]
            CL_command = self.last_u_star[0]
            phi_command = self.last_u_star[1]

            # --- Dynamics Integration ---
            # CRITICAL FIX: Correctly pass CL_command and phi_command
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
                airspeed, w_z, dist_to_thermal, CL_command, phi_command 
            ])

            # Print status every control step
            if self.step_count % self.mpc_update_freq == 0:
                print(f"T={t:.1f}s | Alt={z:.2f}m | Airspeed={airspeed:.2f}m/s | Wz={w_z:.2f}m/s | Dist={dist_to_thermal:.1f}m | CL={CL_command:.2f} | Phi={np.degrees(phi_command):.1f}°")

        self._save_results()

    def _save_results(self):
        """Saves the simulation data to a CSV file."""
        df = pd.DataFrame(self.flight_data, columns=[
            'time', 'x', 'y', 'z', 'Airspeed', 'Wz', 'dist_to_center', 'CL_cmd', 'phi_cmd'
        ])
        
        output_dir = './results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"flight_data_{timestamp}.csv")
        df.to_csv(filepath, index=False)
        
        print(f"\nSimulation results saved to: {filepath}")


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
