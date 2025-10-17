import sys
import os
import numpy as np
import time
import math

# --- CRITICAL FIX: Add 'src' directory to Python path ---
# This guarantees Python can find all custom modules from the 'src' folder.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes 'src' is parallel to 'sim'
src_path = os.path.join(os.path.dirname(current_dir), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)
# --------------------------------------------------------

# The rest of the imports now rely on the corrected sys.path:
from atmospheric_model import AtmosphericModel
from glider_dynamics import GliderDynamics
from mpc_controller import MPCController


# --- SIMULATION CONFIGURATION ---
TIME_STEP = 1.0       # MPC control step (seconds)
SIMULATION_DURATION = 250.0 # Total time to run 
THERMAL_CENTER = np.array([500.0, 500.0]) # Location of the thermal (m)
THERMAL_RADIUS = 100.0 # Thermal radius (m)

# MPC Configuration
MPC_N = 20 # Prediction horizon steps
MPC_DT = TIME_STEP 

# Glider specific parameters
GLIDER_PARAMS = {
    'mass': 700.0, # kg
    'S': 14.0,     # Wing Area (m^2)
    'CD0': 0.015,
    'K': 0.04
}

# Initial State Vector X = [x, y, z, vx, vy, vz, m]
# Starts at 500m high, heading toward the thermal center.
INITIAL_STATE = np.array([20.0, 0.0, 500.0, 20.0, 0.0, -1.0, GLIDER_PARAMS['mass']])


def main_simulation():
    # Adding a clear print statement to confirm the script starts running its logic
    print("--- 🚀 SCRIPT INITIATED SUCCESSFULLY ---") 
    print("--- Bio-Inspired Autonomous Glider Control System Simulation ---")
    print("Setting up simulation components...")

    # 1. Initialize Atmospheric Model (CONFIRMED 2-ARGUMENT CALL)
    atm_model = AtmosphericModel(THERMAL_CENTER, THERMAL_RADIUS)

    # 2. Initialize Glider Dynamics Model
    glider_dynamics = GliderDynamics(GLIDER_PARAMS)

    # 3. Initialize MPC Controller
    mpc_controller = MPCController(N=MPC_N, DT=MPC_DT, glider_params=GLIDER_PARAMS) 

    # --- Setup Complete ---
    current_state = INITIAL_STATE
    t = 0.0
    
    print(f"Thermal at: X={THERMAL_CENTER[0]:.0f}, Y={THERMAL_CENTER[1]:.0f}, Radius: {THERMAL_RADIUS:.0f}m")
    print("Setup complete. Starting simulation loop.")
    
    # Simulation loop
    while t < SIMULATION_DURATION:
        # 1. Sense the Environment
        thermal_params = np.array([THERMAL_CENTER[0], THERMAL_CENTER[1], THERMAL_RADIUS])
        
        # 2. Compute Control Input
        try:
            phi_rad, gamma_rad = mpc_controller.compute_control(current_state, thermal_params)
        except Exception as e:
            # Added more robust logging for silent CasADi failures
            print(f"FATAL MPC SOLVER ERROR at t={t}s: {e}. Using zero control and exiting.")
            phi_rad, gamma_rad = 0.0, 0.0
            # If the solver fails repeatedly, we should terminate
            break


        # 3. Apply Controls and step the dynamics
        wind_vec_3d = atm_model.get_wind_vector(current_state[0], current_state[1], current_state[2])
        
        current_state = glider_dynamics.step_dynamics(
            current_state, 
            phi_rad, 
            gamma_rad, 
            wind_vec_3d, 
            TIME_STEP
        )
        
        t += TIME_STEP

        # 4. Logging and Reporting 
        if int(t) % 10 == 0 or t == TIME_STEP:
            pos_m = current_state[:3]
            V_air_mag = glider_dynamics.calculate_airspeed(current_state, wind_vec_3d)
            
            print(
                f"t={t:.1f}s | "
                f"Pos: ({pos_m[0]:.0f}, {pos_m[1]:.0f}, {pos_m[2]:.0f})m | "
                f"Airspeed: {V_air_mag:.1f}m/s | "
                f"Control (Phi, Gamma): ({np.rad2deg(phi_rad):.1f}°, {np.rad2deg(gamma_rad):.1f}°)"
            )

        # Emergency stop if altitude is dangerously low
        if current_state[2] < 0:
            print(f"\nSimulation terminated early at t={t:.1f}s: Glider crashed (Altitude {current_state[2]:.0f}m).")
            break

    print("\nSimulation complete.")

if __name__ == "__main__":
    main_simulation()