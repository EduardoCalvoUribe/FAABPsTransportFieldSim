import numpy as np
import time
import os

from src.runner import run_payload_simulation
from src.visualization import create_payload_animation


def K_to_c(x1, y1, x2, y2, K):
    """
    Convert standard curvature K=1/R to chord-normalized curvature c.

    Parameters:
    - x1, y1, x2, y2: wall endpoints
    - K: standard curvature (K = 1/R where R is radius)

    Returns:
    - c: chord-normalized curvature parameter

    Example:
    >>> # Wall with radius 50
    >>> wall = [0, 0, 100, 0, K_to_c(0, 0, 100, 0, K=1/50)]
    """
    chord_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return K * chord_length / 2


#####################################################
# HYPERPARAMETERS - Configure everything here       #
#####################################################

# Set random seed for reproducibility
RANDOM_SEED = 42

# Simulation parameters
# Define particle distribution by curvity value: {curvity: count}
# CURVITY_DISTRIBUTION = { # random example
#     -1: 100,
#     -0.4: 300,
#     0: 200,
#     0.6: 100,
#     1: 150,
# }

CURVITY_DISTRIBUTION = {
    -0.3: 300,
}

# Total particles = sum of all counts
N_PARTICLES = sum(CURVITY_DISTRIBUTION.values())

BOX_SIZE = 100
N_STEPS = 2000
SAVE_INTERVAL = 10
DT = 0.01

# Particle parameters
PARTICLE_RADIUS = 1.0
PARTICLE_V0 = 1.0              # Self-propulsion speed
PARTICLE_MOBILITY = 1.0
ROTATIONAL_DIFFUSION = 0.05 #0.05     # Orientational noise

# Payload parameters
PAYLOAD_RADIUS = 10
PAYLOAD_MOBILITY = 1 / PAYLOAD_RADIUS
PAYLOAD_V0 = 0.0 # 0.6 # 0.375                      # Payload self-propulsion speed
PAYLOAD_ROT_DIFFUSION = 0.0 #0.005              # Payload rotational diffusion
PAYLOAD_START_POSITION = np.array([BOX_SIZE/2, BOX_SIZE/2])

# Force parameters
STIFFNESS = 25.0

# Wall configuration (set to None for no walls)
# Walls format: [x1, y1, x2, y2, c]
# where c is chord-normalized curvature:
#   - c = 0: straight line
#   - c = ±1: semicircle
# To use standard curvature K=1/R, use the K_to_c() helper function:
#   - Example: [x1, y1, x2, y2, K_to_c(x1, y1, x2, y2, K=1/50)]
# circle
WALLS = np.array([
    [BOX_SIZE/4, BOX_SIZE/2, 3*BOX_SIZE/4, BOX_SIZE/2, 1],
    [BOX_SIZE/4, BOX_SIZE/2, 3*BOX_SIZE/4, BOX_SIZE/2, -1],
])
WALLS = None

# Example using K (standard curvature):
# R = 100  # radius
# K = 1/R  # curvature
# WALLS = np.array([
#     [0, 0, 200, 0, K_to_c(0, 0, 200, 0, K)],  # curved wall with radius 100
#     [0, 0, 0, 200, 0],  # straight wall
# ])

# WALLS = None


# Visualization parameters
OUTPUT_FILENAME = "D:/PostThesis/visualizations/activepayload2.mp4"           # If None, uses timestamp. Otherwise specify path.
# OUTPUT_FILENAME = "C:/Users/educa/Videos/ye/test.mp4"

# Data saving (set to True to save simulation data)
SAVE_DATA = False
DATA_OUTPUT_PATH = "D:/PostThesis/data/env_wall_test.npz"                    # If None, uses timestamp. Otherwise specify path.


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
        'payload_v0': PAYLOAD_V0,
        'payload_rot_diffusion': PAYLOAD_ROT_DIFFUSION,
        'payload_position': PAYLOAD_START_POSITION,
        'stiffness': STIFFNESS,
        'walls': WALLS if WALLS is not None else np.zeros((0, 5), dtype=np.float64),
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

    # Build curvity array from distribution dictionary
    curvity_array = []
    for curvity_value, count in CURVITY_DISTRIBUTION.items():
        curvity_array.extend([curvity_value] * count)
    curvity_array = np.array(curvity_array)

    params = {
        # Global parameters
        'n_particles': N_PARTICLES,
        'box_size': BOX_SIZE,
        'dt': DT,
        'n_steps': N_STEPS,
        'save_interval': SAVE_INTERVAL,
        'payload_radius': PAYLOAD_RADIUS,
        'payload_mobility': PAYLOAD_MOBILITY,
        'payload_v0': PAYLOAD_V0,
        'payload_rot_diffusion': PAYLOAD_ROT_DIFFUSION,
        'payload_position': PAYLOAD_START_POSITION,
        'stiffness': STIFFNESS,

        # Wall parameters
        'walls': WALLS if WALLS is not None else np.zeros((0, 5), dtype=np.float64),

        # Particle-specific parameters (arrays)
        'v0': np.ones(N_PARTICLES) * PARTICLE_V0,
        'curvity': curvity_array,  # Fixed curvity values from distribution
        'particle_radius': np.ones(N_PARTICLES) * PARTICLE_RADIUS,
        'mobility': np.ones(N_PARTICLES) * PARTICLE_MOBILITY,
        'rot_diffusion': np.ones(N_PARTICLES) * ROTATIONAL_DIFFUSION
    }

    #####################################################
    # RUN SIMULATION                                    #
    #####################################################

    positions, orientations, velocities, payload_positions, payload_velocities, \
    payload_orientations, curvity_values, runtime = run_payload_simulation(params)

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
            payload_orientations, params, curvity_values
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
        positions, orientations, velocities, payload_positions, payload_orientations,
        params, curvity_values, output_file, payload_velocities=payload_velocities
    )

    print("\nPayload simulation and animation completed successfully!")
