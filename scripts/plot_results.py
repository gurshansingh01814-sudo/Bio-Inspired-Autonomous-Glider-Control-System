import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import glob

def find_latest_csv(search_path):
    """Finds the most recently created flight data CSV file."""
    list_of_files = glob.glob(os.path.join(search_path, 'flight_data_*.csv'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def plot_flight_data(filepath):
    """Loads flight data from a CSV and generates key plots."""
    print(f"Loading data from: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return

    # --- Plot 1: Altitude and Thermal Lift ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Altitude (Left Y-axis)
    color = 'tab:blue'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude (m)', color=color)
    ax1.plot(df['time'], df['z'], color=color, label='Altitude (z)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axhline(y=20.0, color='r', linestyle=':', label='Min Alt Constraint (20m)')

    # Thermal Lift (Right Y-axis)
    ax2 = ax1.twinx()  
    color = 'tab:green'
    ax2.set_ylabel('Thermal Lift ($W_z$) (m/s)', color=color)  
    ax2.plot(df['time'], df['Wz'], color=color, linestyle='--', alpha=0.7, label='Thermal Lift')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Combined Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title('Glider Performance: Altitude vs. Thermal Lift')
    plt.show()

    # --- Plot 2: Distance to Thermal Center ---
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['dist_to_center'], label='Distance to Thermal Center', color='tab:purple')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.title('Glider Performance: Target Attraction')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    try:
        # Determine the results directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(current_dir, '..', 'results')
        
        latest_file = find_latest_csv(results_dir)

        if latest_file:
            plot_flight_data(latest_file)
        else:
            print(f"No flight data CSV files found in {results_dir}. Please run main_simulation.py first.")

    except ImportError:
        print("Error: pandas and matplotlib are required for plotting. Please install them:")
        print("pip install pandas matplotlib")
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")