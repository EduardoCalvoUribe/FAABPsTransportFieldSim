import numpy as np
import time
import os

from src.runner import run_payload_simulation
from src.visualization import create_payload_animation, create_averaged_polarity_field_png, create_averaged_polarity_field_by_direction_png


#####################################################
# HYPERPARAMETERS - Configure everything here       #
#####################################################

# Set random seed for reproducibility
RANDOM_SEED = 42

# Simulation parameters
N_PARTICLES = 1200
BOX_SIZE = 300
N_TRAINING_STEPS = 20000  # Number of steps for training phase (circular payload motion)
N_TEST_STEPS = 80000      # Number of steps for test phase (passive payload)
N_STEPS = N_TRAINING_STEPS + N_TEST_STEPS
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
POLARITY_FORCE_SCALING = 0.01  # Scaling factor for force accumulation in polarity fields

# Goal parameters
GOAL_POSITION = np.array([BOX_SIZE*1.125, BOX_SIZE*1.125])  # Bottom-left corner
FORCE_UPDATE_INTERVAL = 20  # How often to store forces in polarity fields (timesteps)
END_WHEN_GOAL_REACHED = False        # If True, simulation ends when payload reaches goal

# Payload circular motion (training phase)
PAYLOAD_CIRCULAR_MOTION = True  # If True, payload follows circular path during training phase
PAYLOAD_CIRCLE_CENTER = np.array([BOX_SIZE/2, BOX_SIZE/2])  # Center of circular path
PAYLOAD_CIRCLE_RADIUS = BOX_SIZE/3  # Radius of circular path
PAYLOAD_N_ROTATIONS = 8  # Number of full rotations during training phase

# Collision-based polarity sharing
COLLISION_SHARE_INTERVAL = 10  # How often to share polarity fields on collision (timesteps, 1=every step)

# Learning control during test phase
TEST_PHASE_LEARNING = 0  # 0=no learning, 1=collision communication only, 2=both F and collision (default)

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
WALLS = None


# Visualization parameters
SHOW_VECTORS = False              # Display polarity vectors as arrows
OUTPUT_FILENAME = "E:/PostThesis/visualizations/test_polfield_collcomms.mp4"           # If None, uses timestamp. Otherwise specify path.
# OUTPUT_FILENAME = "C:/Users/educa/Videos/ye/test_postfield.mp4"

# Data saving (set to True to save simulation data)
SAVE_DATA = False
DATA_OUTPUT_PATH = "E:/PostThesis/data/test_polfield.npz"                    # If None, uses timestamp. Otherwise specify path.


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
        'n_training_steps': 5,
        'save_interval': SAVE_INTERVAL,
        'payload_radius': PAYLOAD_RADIUS,
        'payload_mobility': PAYLOAD_MOBILITY,
        'payload_position': PAYLOAD_START_POSITION,
        'stiffness': STIFFNESS,
        'polarity_force_scaling': POLARITY_FORCE_SCALING,
        'goal_position': GOAL_POSITION,
        'score_and_polarity_update_interval': FORCE_UPDATE_INTERVAL,
        'end_when_goal_reached': END_WHEN_GOAL_REACHED,
        'payload_circular_motion': PAYLOAD_CIRCULAR_MOTION,
        'payload_circle_center': PAYLOAD_CIRCLE_CENTER,
        'payload_circle_radius': PAYLOAD_CIRCLE_RADIUS,
        'payload_n_rotations': PAYLOAD_N_ROTATIONS,
        'collision_share_interval': COLLISION_SHARE_INTERVAL,
        'test_phase_learning': TEST_PHASE_LEARNING,
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
        'n_training_steps': N_TRAINING_STEPS,
        'save_interval': SAVE_INTERVAL,
        'payload_radius': PAYLOAD_RADIUS,
        'payload_mobility': PAYLOAD_MOBILITY,
        'payload_position': PAYLOAD_START_POSITION,
        'stiffness': STIFFNESS,
        'polarity_force_scaling': POLARITY_FORCE_SCALING,

        # Goal parameters
        'goal_position': GOAL_POSITION,
        'score_and_polarity_update_interval': FORCE_UPDATE_INTERVAL,
        'end_when_goal_reached': END_WHEN_GOAL_REACHED,
        'payload_circular_motion': PAYLOAD_CIRCULAR_MOTION,
        'payload_circle_center': PAYLOAD_CIRCLE_CENTER,
        'payload_circle_radius': PAYLOAD_CIRCLE_RADIUS,
        'payload_n_rotations': PAYLOAD_N_ROTATIONS,
        'collision_share_interval': COLLISION_SHARE_INTERVAL,
        'test_phase_learning': TEST_PHASE_LEARNING,

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
    curvity_values, saved_active_polarity, polarity_fields, runtime = run_payload_simulation(params)

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
            params, curvity_values, saved_active_polarity
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
        polarity=saved_active_polarity
    )

    #####################################################
    # CREATE POLARITY FIELD VISUALIZATION               #
    #####################################################

    # Determine polarity field output filenames
    if OUTPUT_FILENAME is None:
        T = int(time.time())
        polarity_field_file_mag = f'./visualizations/polarity_field_T_{T}.png'
        polarity_field_file_dir = f'./visualizations/polarity_field_direction_T_{T}.png'
    else:
        # Use same base name as video but with .png extension
        base_name = OUTPUT_FILENAME.rsplit('.', 1)[0]
        polarity_field_file_mag = f'{base_name}_polarity_field.png'
        polarity_field_file_dir = f'{base_name}_polarity_field_direction.png'

    # Create polarity field visualizations
    create_averaged_polarity_field_png(polarity_fields, BOX_SIZE, polarity_field_file_mag)
    create_averaged_polarity_field_by_direction_png(polarity_fields, BOX_SIZE, polarity_field_file_dir)

    print("\nPayload simulation and animation completed successfully!")
