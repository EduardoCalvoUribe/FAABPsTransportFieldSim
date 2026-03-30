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
N_PARTICLES = 600
BOX_SIZE = 300
N_STEPS = 1000
SAVE_INTERVAL = 10
DT = 0.01

# Particle parameters
PARTICLE_RADIUS = 1.0
PARTICLE_V0 = 5.0              # Self-propulsion speed
PARTICLE_MOBILITY = 1.0
ROTATIONAL_DIFFUSION = 0.2 #0.05     # Orientational noise

MAX_CURVITY = 0.5 #1
MIN_CURVITY = -1.0 #-1
MID_CURVITY = -0.25 #0

# Payload parameters
PAYLOAD_RADIUS = 20
PAYLOAD_MOBILITY = 1 / PAYLOAD_RADIUS
PAYLOAD_START_POSITION = np.array([50.0, 40.0])

# Force parameters
STIFFNESS = 25.0

# Goal parameters
GOAL_POSITION = np.array([150.0, 280.0]) # np.array([83.3, 83.3])  # Top-left corner
PARTICLE_VIEW_RANGE = 0.2 * BOX_SIZE * (1/1.414213)  # Range for goal detection
SCORE_AND_POLARITY_UPDATE_INTERVAL = 20  # How often to update scores & polarity (timesteps)
END_WHEN_GOAL_REACHED = True        # If True, simulation ends when payload reaches goal
POLARITY_NUDGE_INTERVAL = 5        # Every N steps, nudge heading toward polarity
POLARITY_NUDGE_STRENGTH = 0.01       # Angular nudge magnitude (radians)

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
# WALLS = np.array([
#     # Boundary walls
#     [0, 0, 0, BOX_SIZE],
#     [0, 0, BOX_SIZE, 0],
#     [BOX_SIZE, BOX_SIZE, 0, BOX_SIZE],
#     [BOX_SIZE, BOX_SIZE, BOX_SIZE, 0],
#     # Inverted Y shape walls
#     # [2 * BOX_SIZE/6, BOX_SIZE, 4 * BOX_SIZE/6, BOX_SIZE], #top wall
#     [2.2 * BOX_SIZE/6, 4 * BOX_SIZE/7, 2.2 * BOX_SIZE/6, BOX_SIZE], #top left
#     [3.8 * BOX_SIZE/6, 4 * BOX_SIZE/7, 3.8 * BOX_SIZE/6, BOX_SIZE], #top right
#     [2.2 * BOX_SIZE/6, 4 * BOX_SIZE/7, 0, 4 * BOX_SIZE/7], # left shoulder
#     [3.8 * BOX_SIZE/6, 4 * BOX_SIZE/7, BOX_SIZE, 4 * BOX_SIZE/7], # right shoulder
#     # [0, 4 * BOX_SIZE/7, 0, 0], #bot left
#     # [BOX_SIZE, 4*BOX_SIZE/7, BOX_SIZE, 0], #bot right
#     # [0, 0, BOX_SIZE, 0], #bot
#     [2 * BOX_SIZE/7, 2.5 * BOX_SIZE/7, 2 * BOX_SIZE/7, 0], #inner left
#     [2 * BOX_SIZE/7, 2.5 * BOX_SIZE/7, 5 * BOX_SIZE/7, 2.5 * BOX_SIZE/7], #inner top
#     [5 * BOX_SIZE/7, 2.5 * BOX_SIZE/7, 5 * BOX_SIZE/7, 0], #inner right
# ], dtype=np.float64)
# WALLS = np.array([
#     # Boundary walls
#     [0, 0, 0, BOX_SIZE],
#     [0, 0, BOX_SIZE, 0],
#     [BOX_SIZE, BOX_SIZE, 0, BOX_SIZE],
#     [BOX_SIZE, BOX_SIZE, BOX_SIZE, 0],
#     # walls inside
#     # [0, 33.3, 66.6, 33.3],
#     # [33.3, 66.6, 100.0, 66.3]
# ], dtype=np.float64)


# Visualization parameters
SHOW_VECTORS = True              # Display v vectors as arrows
COLOR_BY_SCORE = False           # If True: color by score, if False: color by curvity
OUTPUT_FILENAME = "D:/PostThesis/visualizations/test5_5.mp4"           # If None, uses timestamp. Otherwise specify path.
# OUTPUT_FILENAME = "c:/Users/educa/Downloads/test2.mp4"
# Data saving (set to True to save simulation data)
SAVE_DATA = False
DATA_OUTPUT_PATH = "D:/PostThesis/data/test.npz"                    # If None, uses timestamp. Otherwise specify path.


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
        'end_when_goal_reached': END_WHEN_GOAL_REACHED,
        'polarity_nudge_interval': POLARITY_NUDGE_INTERVAL,
        'polarity_nudge_strength': POLARITY_NUDGE_STRENGTH,
        'walls': WALLS if WALLS is not None else np.zeros((0, 4), dtype=np.float64),
        'v0': np.ones(compile_n_particles) * PARTICLE_V0,
        'curvity': np.zeros(compile_n_particles),
        'particle_radius': np.ones(compile_n_particles) * PARTICLE_RADIUS,
        'mobility': np.ones(compile_n_particles) * PARTICLE_MOBILITY,
        'rot_diffusion': np.ones(compile_n_particles) * ROTATIONAL_DIFFUSION,
        'max_curvity': MAX_CURVITY,
        'min_curvity': MIN_CURVITY,
        'mid_curvity': MID_CURVITY
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
        'end_when_goal_reached': END_WHEN_GOAL_REACHED,
        'polarity_nudge_interval': POLARITY_NUDGE_INTERVAL,
        'polarity_nudge_strength': POLARITY_NUDGE_STRENGTH,

        # Wall parameters
        'walls': WALLS if WALLS is not None else np.zeros((0, 4), dtype=np.float64),

        # Particle-specific parameters (arrays)
        'v0': np.ones(N_PARTICLES) * PARTICLE_V0,
        'curvity': np.zeros(N_PARTICLES),  # Computed dynamically from score & polarity
        'particle_radius': np.ones(N_PARTICLES) * PARTICLE_RADIUS,
        'mobility': np.ones(N_PARTICLES) * PARTICLE_MOBILITY,
        'rot_diffusion': np.ones(N_PARTICLES) * ROTATIONAL_DIFFUSION,

        'max_curvity': MAX_CURVITY,
        'min_curvity': MIN_CURVITY,
        'mid_curvity': MID_CURVITY
    }

    #####################################################
    # RUN SIMULATION                                    #
    #####################################################

    result = run_payload_simulation(params)

    # Unpack results
    (saved_positions, saved_orientations, saved_velocities,
     saved_payload_positions, saved_payload_velocities, saved_curvity,
     saved_polarity, saved_particle_scores, particle_scores, polarity,
     simulation_time, final_step) = result

    print(f"\nSimulation completed in {simulation_time:.2f} seconds")
    print(f"Final step: {final_step}")

    # Save data if enabled
    if SAVE_DATA:
        from src.runner import save_simulation_data
        save_simulation_data(
            DATA_OUTPUT_PATH,
            saved_positions, saved_orientations, saved_velocities,
            saved_payload_positions, saved_payload_velocities,
            params, saved_curvity, saved_polarity, saved_particle_scores
        )
        print(f"Data saved to: {DATA_OUTPUT_PATH}")

    # Create visualization
    create_payload_animation(
        saved_positions, saved_orientations, saved_velocities,
        saved_payload_positions, params, saved_curvity,
        output_file=OUTPUT_FILENAME,
        show_vectors=SHOW_VECTORS,
        polarity=saved_polarity,
        particle_scores=saved_particle_scores if COLOR_BY_SCORE else None
    )

    print("Simulation and visualization completed successfully!")
