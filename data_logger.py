import pandas as pd
import numpy as np

class DataLogger:
    """
    Utility class for logging state, control, and thermal data during the simulation.
    """
    def __init__(self):
        # Initialize an empty list to store dictionary entries for each time step
        self.data_entries = []
        self.columns = [
            'Time', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 
            'phi_cmd', 'gamma_cmd', 'thermal_wz', 'dist_to_thermal'
        ]

    def log_step(self, t, state, control, thermal_wz, dist_to_thermal):
        """
        Logs the state and control at a specific time step.
        """
        # Ensure inputs are flat numpy arrays for consistent logging
        state = np.array(state).flatten()
        control = np.array(control).flatten()
        
        entry = {
            'Time': t,
            'x': state[0],
            'y': state[1],
            'z': state[2],
            'vx': state[3],
            'vy': state[4],
            'vz': state[5],
            'm': state[6],
            'phi_cmd': np.rad2deg(control[0]), # Log in degrees for readability
            'gamma_cmd': np.rad2deg(control[1]),
            'thermal_wz': thermal_wz,
            'dist_to_thermal': dist_to_thermal
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
        
        # --- Print Simulation Summary ---
        print("\n" + "="*80)
        print("                       GLIDER SIMULATION FLIGHT SUMMARY")
        print("="*80)
        
        # Key performance indicators
        final_z = df['z'].iloc[-1]
        max_z = df['z'].max()
        min_z = df['z'].min()
        
        print(f"Total Simulation Time: {df['Time'].iloc[-1]:.1f} seconds")
        print(f"Initial Altitude (m): {df['z'].iloc[0]:.2f}")
        print(f"Final Altitude (m): {final_z:.2f}")
        print(f"Max Altitude Reached (m): {max_z:.2f}")
        print(f"Min Altitude (m): {min_z:.2f}")
        print(f"Total Altitude Change (m): {final_z - df['z'].iloc[0]:.2f}")
        print("-" * 80)
        
        # Print the first and last 5 steps for quick inspection
        print("Sampled Trajectory Data (First 5 Steps):")
        print(df.head().to_markdown(index=False, floatfmt=".2f"))
        print("\nSampled Trajectory Data (Last 5 Steps):")
        print(df.tail().to_markdown(index=False, floatfmt=".2f"))
        print("="*80)
        
        # For deeper debugging, save the full dataframe to an instance variable
        self.full_data = df
