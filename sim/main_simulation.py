import numpy as np
import time
import os
import pandas as pd
import sys

# Assume the necessary classes are now correctly imported from src/
# Adjust imports based on your actual project structure if necessary
try:
    from src.glider_dynamics import GliderDynamics
    from src.mpc_controller import MPCController
    from src.atmospheric_model import AtmosphericModel
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class GliderControlSystem:
    """
    Manages the complete simulation, integrating the glider dynamics, 
    the atmospheric model, and the MPC controller.
    """

    def __init__(self, config_path='./data/glider_config.yaml'):
        # Initialize sub-systems
        self.config_path = config_path
        self.glider = GliderDynamics(config_path)
        self.thermal = AtmosphericModel(config_path)
        self.mpc = MPCController(config_path)
        
        # Simulation parameters (from config or defaults)
        self.T_end = 200.0  # Total simulation time
        self.dt_sim = 0.5   # Simulation time step (for integration)
        self.dt_mpc = self.mpc.DT # MPC control frequency (2.0s)
        self.T_end = 200.0
        
        # Calculate maximum number of steps
        self.max_steps = int(self.T_end / self.dt_sim)
        self.mpc_update_freq = int(self.dt_mpc / self.dt_sim)
        
        self.flight_data = []
        print("System initialized successfully.")

    def main_loop(self):
        """Runs the main time-stepping simulation."""
        
        print("Starting simulation loop...")
        
        T_end = self.T_end
        dt_sim = self.dt_sim
        current_state = self.glider.get_state()
        t = 0.0
        
        # Placeholder for previous control inputs (used if IPOPT fails)
        last_u_star = np.array([0.0, 0.0])

        while t < T_end:
            
            # Unpack state
            x, y, z, vx, vy, vz = current_state
            
            # Check for crash condition
            if z <= self.mpc.config.get('CONSTRAINTS', {}).get('Z_MIN', 20.0):
                print(f"\n--- LANDING / CRASH: Minimum altitude reached. Simulation terminated. ---")
                break
                
            # --- MPC Control Update (Run every dt_mpc seconds) ---
            if (t * 100) % (self.dt_mpc * 100) < (self.dt_sim * 100):
                
                thermal_center = np.array([self.thermal.center_x, self.thermal.center_y])
                thermal_radius = self.thermal.radius
                
                # Wz prediction (Simplified: Assume Wz is constant 0 across the horizon for now)
                Wz_prediction = np.zeros((1, self.mpc.N)) 

                # CRITICAL FIX: Changed method name to the correct solve_mpc
                u_star = self.mpc.solve_mpc(current_state, thermal_center, Wz_prediction) 
                
                # Check if solver failed (solve_mpc returns [0, 0] on failure)
                if np.allclose(u_star, np.array([0.0, 0.0])):
                    phi_command, alpha_command = last_u_star
                else:
                    phi_command, alpha_command = u_star[0], u_star[1]
                    last_u_star = u_star
            else:
                # Maintain last commanded input if not in MPC step
                phi_command, alpha_command = last_u_star

            # --- Dynamics Integration (Run every dt_sim seconds) ---
            new_state = self.glider.update(phi_command, alpha_command, dt_sim, self.thermal)
            current_state = new_state
            t += dt_sim
            
            # Calculate metrics
            airspeed = np.linalg.norm(current_state[3:6])
            w_z = self.thermal.get_thermal_lift(current_state[0], current_state[1], current_state[2])
            dist_to_thermal = np.linalg.norm(current_state[0:2] - np.array([self.thermal.center_x, self.thermal.center_y]))
            
            # Log data
            self.flight_data.append([
                t, current_state[0], current_state[1], current_state[2], 
                airspeed, w_z, dist_to_thermal, phi_command, alpha_command
            ])

            # Print status every 2.0 seconds
            if (t * 10) % 20 == 0:
                print(f"T={t:.1f}s | Alt={z:.2f}m | Airspeed={airspeed:.2f}m/s | Wz={w_z:.2f}m/s | Dist={dist_to_thermal:.1f}m")

        self._save_results()

    def _save_results(self):
        """Saves the simulation data to a CSV file."""
        df = pd.DataFrame(self.flight_data, columns=[
            'Time', 'X', 'Y', 'Z', 'Airspeed', 'Wz_Atmosphere', 'Dist_to_Thermal', 'Phi_Command', 'Alpha_Command'
        ])
        
        output_dir = './results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"flight_data_{timestamp}.csv")
        df.to_csv(filepath, index=False)
        
        print(f"\nSimulation results saved to: {filepath}")
        print("Simulation successful. Run 'python scripts/plot_results.py' to visualize the flight data stored in the file.")


if __name__ == '__main__':
    # Determine the configuration path relative to the current working directory
    # Assumes config file is always available relative to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'data', 'glider_config.yaml')
    
    # Initialize and run the system
    system = GliderControlSystem(config_path)
    system.main_loop()
