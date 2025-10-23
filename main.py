import numpy as np
import time
import os

from src.runner import run_payload_simulation
from src.visualization import create_payload_animation


#####################################################
# HYPERPARAMETERS - Configure everything here       #
#####################################################

# Set random seed for reproducibility
RANDOM_SEED = 42

# Simulation parameters
N_PARTICLES = 1200
BOX_SIZE = 300
N_STEPS = 300000
SAVE_INTERVAL = 10
DT = 0.01

# Particle parameters
PARTICLE_RADIUS = 1.0
PARTICLE_V0 = 3.75              # Self-propulsion speed
PARTICLE_MOBILITY = 1.0
ROTATIONAL_DIFFUSION = 0.05     # Orientational noise

# Payload parameters
PAYLOAD_RADIUS = 20
PAYLOAD_MOBILITY = 1 / PAYLOAD_RADIUS
PAYLOAD_START_POSITION = np.array([BOX_SIZE/6, 5 * BOX_SIZE/6])  # Top-left corner

# Force parameters
STIFFNESS = 25.0

# Goal parameters
GOAL_POSITION = np.array([BOX_SIZE*0.125, BOX_SIZE*0.125])  # Bottom-left corner
PARTICLE_VIEW_RANGE = 0.1 * BOX_SIZE  # Range for goal detection
SCORE_AND_POLARITY_UPDATE_INTERVAL = 50  # How often to update scores & polarity (timesteps)
DIRECTEDNESS = 1                    # 0 = pure vicsek alignment, 1 = pure gradient following
END_WHEN_GOAL_REACHED = True        # If True, simulation ends when payload reaches goal

# Wall configuration (set to None for no walls)
# Example walls:
WALLS = np.array([
    # Boundary walls
    [0, 0, 0, BOX_SIZE],
    [0, 0, BOX_SIZE, 0],
    [BOX_SIZE, BOX_SIZE, 0, BOX_SIZE],
    [BOX_SIZE, BOX_SIZE, BOX_SIZE, 0],
    # Maze walls
    # [BOX_SIZE*0.33, BOX_SIZE*0.66, BOX_SIZE, BOX_SIZE*0.66],
    [0, BOX_SIZE*0.25, BOX_SIZE*0.55, BOX_SIZE*0.25], # bottom wall
    [BOX_SIZE*0.375, BOX_SIZE, BOX_SIZE*0.375, BOX_SIZE*0.45], # top left wall
    [BOX_SIZE*0.75, BOX_SIZE, BOX_SIZE*0.75, BOX_SIZE*0.45], # top right wall
], dtype=np.float64)
# WALLS = None


# Visualization parameters
SHOW_VECTORS = False              # Display v vectors as arrows
COLOR_BY_SCORE = False           # If True: color by score, if False: color by curvity
OUTPUT_FILENAME = "E:/PostThesis/visualizations/polarity_test_deadend.mp4"           # If None, uses timestamp. Otherwise specify path.

# Data saving (set to True to save simulation data)
SAVE_DATA = False
DATA_OUTPUT_PATH = "E:/PostThesis/data/polarity_test_deadend.npz"                    # If None, uses timestamp. Otherwise specify path.


#####################
# Main execution    #
#####################

if __name__ == "__main__":

    # Set random seed
    np.random.seed(RANDOM_SEED)

    # Create directories if they don't exist
    if SAVE_DATA:
        os.makedirs('./data', exist_ok=True)
    os.makedirs('./visualizations', exist_ok=True)

    #####################################################
    # JIT COMPILATION                                   #
    #####################################################

    print("Compiling JIT functions...")

    # Build parameter dictionary for compilation run
    compile_n_particles = 10
    compile_params = {
        'n_particles': compile_n_particles,
        'box_size': BOX_SIZE,
        'dt': DT,
        'n_steps': 10,
        'save_interval': SAVE_INTERVAL,
        'payload_radius': PAYLOAD_RADIUS,
        'payload_mobility': PAYLOAD_MOBILITY,
        'payload_position': PAYLOAD_START_POSITION,
        'stiffness': STIFFNESS,
        'goal_position': GOAL_POSITION,
        'particle_view_range': PARTICLE_VIEW_RANGE,
        'score_and_polarity_update_interval': SCORE_AND_POLARITY_UPDATE_INTERVAL,
        'directedness': DIRECTEDNESS,
        'end_when_goal_reached': END_WHEN_GOAL_REACHED,
        'walls': WALLS if WALLS is not None else np.zeros((0, 4), dtype=np.float64),
        'v0': np.ones(compile_n_particles) * PARTICLE_V0,
        'curvity': np.zeros(compile_n_particles),
        'particle_radius': np.ones(compile_n_particles) * PARTICLE_RADIUS,
        'mobility': np.ones(compile_n_particles) * PARTICLE_MOBILITY,
        'rot_diffusion': np.ones(compile_n_particles) * ROTATIONAL_DIFFUSION
    }

    run_payload_simulation(compile_params)
    print("JIT compilation complete.\n")

    #####################################################
    # BUILD SIMULATION PARAMETERS                       #
    #####################################################

    params = {
        # Global parameters
        'n_particles': N_PARTICLES,
        'box_size': BOX_SIZE,
        'dt': DT,
        'n_steps': N_STEPS,
        'save_interval': SAVE_INTERVAL,
        'payload_radius': PAYLOAD_RADIUS,
        'payload_mobility': PAYLOAD_MOBILITY,
        'payload_position': PAYLOAD_START_POSITION,
        'stiffness': STIFFNESS,

        # Goal parameters
        'goal_position': GOAL_POSITION,
        'particle_view_range': PARTICLE_VIEW_RANGE,
        'score_and_polarity_update_interval': SCORE_AND_POLARITY_UPDATE_INTERVAL,
        'directedness': DIRECTEDNESS,
        'end_when_goal_reached': END_WHEN_GOAL_REACHED,

        # Wall parameters
        'walls': WALLS if WALLS is not None else np.zeros((0, 4), dtype=np.float64),

        # Particle-specific parameters (arrays)
        'v0': np.ones(N_PARTICLES) * PARTICLE_V0,
        'curvity': np.zeros(N_PARTICLES),  # Computed dynamically from score & polarity
        'particle_radius': np.ones(N_PARTICLES) * PARTICLE_RADIUS,
        'mobility': np.ones(N_PARTICLES) * PARTICLE_MOBILITY,
        'rot_diffusion': np.ones(N_PARTICLES) * ROTATIONAL_DIFFUSION
    }

    #####################################################
    # RUN SIMULATION                                    #
    #####################################################

    positions, orientations, velocities, payload_positions, payload_velocities, \
    curvity_values, saved_polarity, saved_particle_scores, \
    particle_scores, polarity, runtime = run_payload_simulation(params)

    #####################################################
    # SAVE DATA (optional)                              #
    #####################################################

    if SAVE_DATA:
        from src.runner import save_simulation_data
        # Determine data output filename
        if DATA_OUTPUT_PATH is None:
            T = int(time.time())
            data_file = f'./data/sim_data_T_{T}.npz'
        else:
            data_file = DATA_OUTPUT_PATH
        save_simulation_data(
            data_file,
            positions, orientations, velocities, payload_positions, payload_velocities,
            params, curvity_values, saved_polarity, saved_particle_scores
        )

    #####################################################
    # CREATE ANIMATION                                  #
    #####################################################

    # Determine output filename
    if OUTPUT_FILENAME is None:
        T = int(time.time())
        output_file = f'./visualizations/sim_animation_T_{T}.mp4'
    else:
        output_file = OUTPUT_FILENAME

    # Create animation
    create_payload_animation(
        positions, orientations, velocities, payload_positions, params,
        curvity_values, output_file,
        show_vectors=SHOW_VECTORS,
        polarity=saved_polarity,
        particle_scores=saved_particle_scores if COLOR_BY_SCORE else None
    )

    print("\nPayload simulation and animation completed successfully!")
