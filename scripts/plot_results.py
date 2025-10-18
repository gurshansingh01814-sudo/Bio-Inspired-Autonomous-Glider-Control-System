import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # REQUIRED for 3D plotting
import sys
import os
import glob
import numpy as np

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
    
    # Check if necessary columns exist (assuming data logging is correct now)
    if 'CL_cmd' not in df.columns or 'phi_cmd' not in df.columns:
        print("Warning: Missing control command columns (CL_cmd, phi_cmd). Skipping control plots.")
        # Fill with zeros or a constant if missing for robustness
        df['CL_cmd'] = df.get('CL_cmd', np.nan)
        df['phi_cmd'] = df.get('phi_cmd', np.nan)
    
    # Calculate horizontal distance for convenience (assumes thermal center is at 500, 500 for visualization)
    # The actual thermal center should be retrieved from the config/logged data if possible.
    THERMAL_X, THERMAL_Y = 500.0, 500.0 
    
    # -----------------------------------------------------------------
    # --- Plot 1: Altitude and Thermal Lift (Original Dual-Axis) ---
    # -----------------------------------------------------------------
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
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.title('Glider Performance: Altitude, Climbing, and Thermal Lift Over Time')
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------
    # --- Plot 2: 3D Trajectory (New Critical Plot) ---
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot glider path
    ax.plot(df['x'], df['y'], df['z'], label='Glider Path', color='tab:red', linewidth=1.5)
    
    # Scatter plot of thermal center (assuming constant height for visualization)
    ax.scatter([THERMAL_X], [THERMAL_Y], [df['z'].min()], color='k', marker='^', s=100, label='Thermal Center (Ground)')
    
    # Plot a cylinder/circle representing the thermal radius
    thermal_radius = 150.0 # Should ideally be loaded from config
    t = np.linspace(0, 2*np.pi, 100)
    ax.plot(THERMAL_X + thermal_radius*np.cos(t), 
            THERMAL_Y + thermal_radius*np.sin(t), 
            df['z'].min(), 'k--', alpha=0.5, label='Thermal Boundary')
            
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Altitude (m)')
    ax.set_title('3D Flight Trajectory and Thermal Interaction')
    ax.legend()
    ax.view_init(elev=25., azim=-150) # Set a good viewing angle
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------
    # --- Plot 3: Control Inputs (New Critical Plot for MPC analysis) ---
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top Plot: Lift Coefficient (CL)
    axes[0].plot(df['time'], df['CL_cmd'], label='$C_L$ Command', color='tab:orange')
    axes[0].set_ylabel('Lift Coefficient ($C_L$)', fontsize=12)
    axes[0].axhline(y=1.2, color='r', linestyle=':', alpha=0.7) # CL_MAX
    axes[0].axhline(y=0.2, color='r', linestyle=':', alpha=0.7) # CL_MIN
    axes[0].set_title('MPC Control Input Analysis')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Bottom Plot: Bank Angle (Phi)
    # Convert radians to degrees for easier interpretation
    axes[1].plot(df['time'], np.degrees(df['phi_cmd']), label='Bank Angle ($\phi$)', color='tab:brown')
    axes[1].set_ylabel('Bank Angle ($\phi$) (deg)', fontsize=12)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].axhline(y=45.0, color='r', linestyle=':', alpha=0.7) # Max Bank
    axes[1].axhline(y=-45.0, color='r', linestyle=':', alpha=0.7) # Min Bank
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    try:
        # Determine the results directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Assuming script is in 'scripts/' and results in 'results/'
        results_dir = os.path.join(current_dir, '..', 'results') 
        
        latest_file = find_latest_csv(results_dir)

        if latest_file:
            plot_flight_data(latest_file)
        else:
            print(f"No flight data CSV files found in {results_dir}. Please run main_simulation.py first.")

    except ImportError as e:
        print("Error: Required libraries (pandas, matplotlib, numpy) are missing.")
        print("Please install them using: pip install pandas matplotlib numpy")
        print(f"Detailed Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")