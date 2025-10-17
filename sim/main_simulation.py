import sys
import os

# --- CRITICAL FIX START ---
# Get the path of the project root directory (one level up from sim/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the project root directory to the Python search path so it can find 'mpc_controller'
if parent_dir not in sys.path:
    sys.path.append(parent_dir) 
# --- CRITICAL FIX END ---

from mpc_controller import MPCController
from data_logger import DataLogger
import numpy as np
import math

class GliderDynamics:
    """Simplified Glider Point Mass Dynamics."""
    def __init__(self, params):
        self.m = params.get('mass', 700.0)
        self.g = 9.81
        self.rho = 1.225
        self.S = params.get('S', 14.0)
        self.CD0 = params.get('CD0', 0.015)
        self.K = params.get('K', 0.04)

        # Thermal properties (used here for simulation truth, not MPC knowledge)
        self.thermal_cx = 50.0
        self.thermal_cy = 50.0
        self.thermal_radius = 200.0
        self.W_z_max = 5.0
        self.CL = 0.8 # Constant CL for simulation

    def get_thermal_uplift(self, x, y):
        """Calculates vertical wind speed based on location."""
        dist = np.sqrt((x - self.thermal_cx)**2 + (y - self.thermal_cy)**2)
        
        # Use a smooth cosine model (same as in MPC)
        if dist < self.thermal_radius:
            dist_ratio = dist / self.thermal_radius
            # Smooth function: goes from W_z_max at center to 0 at radius
            W_z = self.W_z_max * (np.cos(np.pi * dist_ratio) + 1.0) / 2.0
        else:
            W_z = 0.0
        return W_z, dist

    def step(self, state, control, dt):
        """
        Integrates the 7-state dynamics (Euler integration).
        state = [x, y, z, vx, vy, vz, m]
        control = [phi, gamma]
        """
        x, y, z, vx, vy, vz, m = state
        phi, gamma = control

        # --- 1. Environmental Winds ---
        W_atm_z, dist_to_thermal = self.get_thermal_uplift(x, y)
        W_atm_vec = np.array([0.0, 0.0, W_atm_z])
        V_ground_vec = np.array([vx, vy, vz])

        # --- 2. Airspeed Vector ---
        V_air_vec = V_ground_vec - W_atm_vec
        V_air = np.linalg.norm(V_air_vec)
        
        # Guard against zero airspeed (using a minimum operational check)
        if V_air < 1e-3:
            V_air = 1e-3 # Prevent division by zero, though MPC should prevent this
        
        e_v = V_air_vec / V_air # Unit vector of V_air (direction of relative wind)

        # --- 3. Aerodynamic Forces ---
        CD = self.CD0 + self.K * self.CL**2
        
        L_mag = 0.5 * self.rho * self.S * self.CL * V_air**2
        D_mag = 0.5 * self.rho * self.S * CD * V_air**2
        
        # Drag Force opposes the relative wind direction
        D_vec = -D_mag * e_v
        
        # Lift Force (Simplified projection using bank/pitch control, same as in MPC)
        # Note: This projection is highly simplified to align with the MPC's CasADi model
        L_x = L_mag * (np.sin(phi) * np.sin(gamma)) 
        L_y = L_mag * (-np.cos(phi) * np.sin(gamma)) 
        L_z = L_mag * (np.cos(gamma)) 
        L_vec = np.array([L_x, L_y, L_z])

        # --- 4. Total Force & Acceleration ---
        G_vec = np.array([0.0, 0.0, -m * self.g])
        F_total = L_vec + D_vec + G_vec
        a_vec = F_total / m

        # --- 5. Integration (Euler Step) ---
        x_dot = np.concatenate([V_ground_vec, a_vec, [0.0]]) # Mass derivative is 0
        next_state = state + dt * x_dot
        
        # Apply ground constraint 
        if next_state[2] < 0.0:
            next_state[2] = 0.0
            next_state[5] = 0.0
            
        return next_state, W_atm_z, dist_to_thermal

def run_simulation():
    """Main simulation loop: Initializes components and drives the simulation forward."""
    
    # ------------------ Initialization ------------------
    N = 20          # MPC Horizon steps
    DT = 1.0        # MPC/Simulation Time Step (seconds)
    SIM_TIME = 120  # Total simulation time
    TOTAL_STEPS = int(SIM_TIME / DT)

    glider_params = {
        'mass': 700.0, 
        'S': 14.0,
        'CD0': 0.015,
        'K': 0.04
    }

    # Initial State: [x, y, z, vx, vy, vz, m]
    initial_state = np.array([0.0, 0.0, 150.0, 15.0, 0.0, -1.0, 700.0]) 

    # Thermal Parameters: [cx, cy, radius]
    thermal_params = np.array([50.0, 50.0, 200.0]) 
    
    # Instantiate Components
    glider_dynamics = GliderDynamics(glider_params)
    mpc = MPCController(N=N, DT=DT, glider_params=glider_params)
    logger = DataLogger()

    # Initial control guess 
    current_control = np.array([0.0, 0.0])
    
    # ------------------ Simulation Loop ------------------
    print(f"Starting Glider Simulation for {SIM_TIME} seconds...")
    sys.stdout.flush()
    current_state = initial_state
    
    for k in range(TOTAL_STEPS):
        t = k * DT
        
        # --- 1. Compute Control ---
        control_action = mpc.compute_control(current_state, thermal_params)
        current_control = control_action 
        
        # --- 2. Step Dynamics ---
        next_state, W_atm_z, dist_to_thermal = glider_dynamics.step(current_state, current_control, DT)
        current_state = next_state
        
        # --- 3. Log Data ---
        logger.log_step(t + DT, current_state, current_control, W_atm_z, dist_to_thermal)
        
        # Termination Condition
        if current_state[2] < 5.0 and k > 0:
            print(f"Altitude too low. Terminating simulation at t={t+DT:.1f}s.")
            sys.stdout.flush()
            break
            
        # Feedback print every 10 steps
        if k % 10 == 0:
            print(f"T={t+DT:.1f}s, Alt={current_state[2]:.2f}m, V_air (Approx)={np.linalg.norm(current_state[3:6]):.2f}m/s, Wz={W_atm_z:.2f}m/s")
            sys.stdout.flush() 
    
    # ------------------ Results ------------------
    logger.print_summary()


if __name__ == '__main__':
    run_simulation()
