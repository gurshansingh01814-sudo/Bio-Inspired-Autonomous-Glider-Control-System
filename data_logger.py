import pandas as pd
import numpy as np

class DataLogger:
    """
    Utility class for logging state, control, and thermal data during the simulation.
    """
    def __init__(self):
        # Initialize an empty list to store dictionary entries for each time step
        self.data_entries = []
        # CRITICAL FIX: Columns aligned with 6D state and [CL, Phi] control
        self.columns = [
            'time', 'x', 'y', 'z', 'vx', 'vy', 'vz', 
            'CL_cmd', 'phi_cmd', 'Wz', 'dist_to_center'
        ]

    def log_step(self, t, state, control, thermal_wz, dist_to_thermal):
        """
        Logs the state and control at a specific time step.
        Control input 'control' is [CL_command, phi_command (rad)]
        """
        state = np.array(state).flatten()
        control = np.array(control).flatten()
        
        if len(state) != 6:
             raise ValueError(f"State vector must be 6D, but got {len(state)} dimensions.")
        
        entry = {
            'time': t,
            'x': state[0],
            'y': state[1],
            'z': state[2],
            'vx': state[3],
            'vy': state[4],
            'vz': state[5],
            
            # CRITICAL FIX: Log CL directly (dimensionless coefficient)
            'CL_cmd': control[0], 
            
            # CRITICAL FIX: Log Phi in radians to match the simulator
            'phi_cmd': control[1], 
            
            # CRITICAL FIX: Consistent naming with main_simulation and plotter
            'Wz': thermal_wz, 
            'dist_to_center': dist_to_thermal
        }
        self.data_entries.append(entry)

    def print_summary(self):
        """
        Creates a pandas DataFrame and prints a summary of the flight.
        """
        if not self.data_entries:
            print("\n--- Simulation Summary ---")
            print("No data logged. Simulation likely crashed immediately.")
            return

        df = pd.DataFrame(self.data_entries)
        
        print("\n" + "="*80)
        print("                   BIO-INSPIRED GLIDER FLIGHT SUMMARY")
        print("="*80)
        
        final_z = df['z'].iloc[-1]
        max_z = df['z'].max()
        
        print(f"Total Simulation Time: {df['time'].iloc[-1]:.1f} seconds")
        print(f"Max Altitude Reached (m): {max_z:.2f} 🚀")
        print(f"Total Altitude Change (m): {final_z - df['z'].iloc[0]:.2f}")
        print("-" * 80)
        
        df_summary = df.copy()
        if 'phi_cmd' in df_summary.columns:
             df_summary['phi_cmd (deg)'] = np.degrees(df_summary['phi_cmd'])
             df_summary = df_summary.drop(columns=['phi_cmd'])
        
        print("Sampled Trajectory Data (Last 5 Steps):")
        print(df_summary.tail().to_markdown(index=False, floatfmt=".2f"))
        print("="*80)
        
        self.full_data = df