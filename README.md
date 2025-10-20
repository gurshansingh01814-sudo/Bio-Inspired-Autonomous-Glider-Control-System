# Bio-Inspired-Autonomous-Glider-Control-System 
# Project Overview-
         This project implements a fully non-linear Model Predictive Control (MPC) system to enable an autonomous glider to locate, capture, and climb within thermal updrafts—a critical            bio-inspired strategy used by real-world soaring birds. The system successfully demonstrates closed-loop control over a 6-Degrees-of-Freedom (DOF) glider model, optimizing lift               coefficient ($\mathbf{C_L}$) and bank angle ($\mathbf{\phi}$) to maximize altitude gain.
# Installation and Setup- 
             This project requires Python and specialized libraries for dynamics, control, and optimization
             Prerequisites 
             All required Python libraries and their versions are listed in requirements.txt
             Setup Instructions
                Clone the repository: git clone [gh repo clone gurshansingh01814-sudo/Bio-Inspired-Autonomous-Glider-Control-System]
                                      cd bio-inspired-autonomous-glider-control
                Install dependencies: Use the provided requirements file : pip install -r requirements.txt
# Running the Simulation-
            Configure the Run-
            Adjust the flight parameters, initial state, and thermal environment in the configuration file:
                                       data/glider_config.yaml: Modify the initial_state, thermal location, and control constraints here.
            Execute the Simulation-
            Run the main simulation loop. The data_logger.py module will automatically record data points at set intervals.
                                       python -m sim.main_simulation
            Generate Results and Plots-
            After the simulation finishes, run the plotting script to visualize the final performance and control decisions.
                                       python scripts/plot_results
# Project Structure-
                     File/Folder                                      Description
                     sim.main_simulation.py                           Main entry point; loads configuration, initializes components, and executes the control loop.
                     requirements.txt                                 Lists all necessary Python dependencies (NumPy, CasADi, etc.)
                     project_documentation.md                         Detailed technical report, including mathematical models, MPC formulation, and real-world applications (Essential 
                                                                      reading for technical review)
                     src/glider_dynamics.py                           src/glider_dynamics.py
                     src/mpc_controller.py                            Defines the MPC optimization problem (CasADi), objective, and constraints.
                     src/atmospheric_model.py                         Calculates the thermal lift ($\mathbf{W_z}$) based on location.
                     data_logger.py                                   Modular class for efficient and structured recording of all simulation and control data
                     
