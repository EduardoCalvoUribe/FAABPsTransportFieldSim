import numpy as np
import time
import os

from src.runner import run_hollow_payload_simulation
from src.visualization import create_hollow_payload_animation


def K_to_c(x1, y1, x2, y2, K):
    """
    Convert standard curvature K=1/R to chord-normalized curvature c.

    Parameters:
    - x1, y1, x2, y2: wall endpoints
    - K: standard curvature (K = 1/R where R is radius)

    Returns:
    - c: chord-normalized curvature parameter
    """
    chord_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return K * chord_length / 2


#####################################################
# HYPERPARAMETERS - Configure everything here       #
#####################################################

# Set random seed for reproducibility
RANDOM_SEED = 42

# Simulation parameters
CURVITY_DISTRIBUTION = {
    0.1: 100,
}

# Total particles = sum of all counts
N_PARTICLES = sum(CURVITY_DISTRIBUTION.values())

BOX_SIZE = 100
N_STEPS = 20000
SAVE_INTERVAL = 10
DT = 0.01

# Particle parameters
PARTICLE_RADIUS = 1.0
PARTICLE_V0 = 3.75              # Self-propulsion speed
PARTICLE_MOBILITY = 1.0
ROTATIONAL_DIFFUSION = 0.05        # Orientational noise

# Hollow payload parameters
# The hollow payload is a ring with inner and outer radius
# Particles inside the hollow center can push outward
# Particles outside can push inward
PAYLOAD_INNER_RADIUS = 24       # Inner radius (the hole)
PAYLOAD_OUTER_RADIUS = 25       # Outer radius
PAYLOAD_INNER_OFFSET = np.array([0.0, 0.0])  # Offset of inner circle from outer circle center
PAYLOAD_MOBILITY = 1 / PAYLOAD_OUTER_RADIUS  # Based on outer radius
PAYLOAD_START_POSITION = np.array([BOX_SIZE/2, BOX_SIZE/2])

# Force parameters
STIFFNESS = 25.0

# Wall configuration (set to None for no walls)
WALLS = None
# Example: boundary walls
# WALLS = np.array([
#     [0, 0, 0, BOX_SIZE, 0],
#     [0, 0, BOX_SIZE, 0, 0],
#     [BOX_SIZE, BOX_SIZE, 0, BOX_SIZE, 0],
#     [BOX_SIZE, BOX_SIZE, BOX_SIZE, 0, 0],
# ], dtype=np.float64)

# Visualization parameters
OUTPUT_FILENAME = None  # If None, uses timestamp

# Data saving
SAVE_DATA = False
DATA_OUTPUT_PATH = None


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

    print("Compiling JIT functions for hollow payload...")

    # Build parameter dictionary for compilation run
    compile_n_particles = 10
    compile_params = {
        'n_particles': compile_n_particles,
        'box_size': BOX_SIZE,
        'dt': DT,
        'n_steps': 10,
        'save_interval': SAVE_INTERVAL,
        'payload_inner_radius': PAYLOAD_INNER_RADIUS,
        'payload_outer_radius': PAYLOAD_OUTER_RADIUS,
        'payload_inner_offset': PAYLOAD_INNER_OFFSET,
        'payload_mobility': PAYLOAD_MOBILITY,
        'payload_position': PAYLOAD_START_POSITION,
        'stiffness': STIFFNESS,
        'walls': WALLS if WALLS is not None else np.zeros((0, 5), dtype=np.float64),
        'v0': np.ones(compile_n_particles) * PARTICLE_V0,
        'curvity': np.zeros(compile_n_particles),
        'particle_radius': np.ones(compile_n_particles) * PARTICLE_RADIUS,
        'mobility': np.ones(compile_n_particles) * PARTICLE_MOBILITY,
        'rot_diffusion': np.ones(compile_n_particles) * ROTATIONAL_DIFFUSION
    }

    run_hollow_payload_simulation(compile_params)
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
        'n_particles_inside': 100,  # First 300 particles go inside
        'box_size': BOX_SIZE,
        'dt': DT,
        'n_steps': N_STEPS,
        'save_interval': SAVE_INTERVAL,
        'payload_inner_radius': PAYLOAD_INNER_RADIUS,
        'payload_outer_radius': PAYLOAD_OUTER_RADIUS,
        'payload_inner_offset': PAYLOAD_INNER_OFFSET,
        'payload_mobility': PAYLOAD_MOBILITY,
        'payload_position': PAYLOAD_START_POSITION,
        'stiffness': STIFFNESS,

        # Wall parameters
        'walls': WALLS if WALLS is not None else np.zeros((0, 5), dtype=np.float64),

        # Particle-specific parameters (arrays)
        'v0': np.ones(N_PARTICLES) * PARTICLE_V0,
        'curvity': curvity_array,
        'particle_radius': np.ones(N_PARTICLES) * PARTICLE_RADIUS,
        'mobility': np.ones(N_PARTICLES) * PARTICLE_MOBILITY,
        'rot_diffusion': np.ones(N_PARTICLES) * ROTATIONAL_DIFFUSION
    }

    #####################################################
    # RUN SIMULATION                                    #
    #####################################################

    positions, orientations, velocities, payload_positions, payload_velocities, \
    curvity_values, runtime = run_hollow_payload_simulation(params)

    #####################################################
    # SAVE DATA (optional)                              #
    #####################################################

    if SAVE_DATA:
        from src.runner import save_simulation_data
        if DATA_OUTPUT_PATH is None:
            T = int(time.time())
            data_file = f'./data/hollow_payload_data_T_{T}.npz'
        else:
            data_file = DATA_OUTPUT_PATH
        # Note: save_simulation_data would need modification for hollow payload params
        # For now, we just save the basic data
        np.savez(
            data_file,
            positions=positions,
            orientations=orientations,
            velocities=velocities,
            payload_positions=payload_positions,
            payload_velocities=payload_velocities,
            curvity_values=curvity_values,
            payload_inner_radius=PAYLOAD_INNER_RADIUS,
            payload_outer_radius=PAYLOAD_OUTER_RADIUS,
            payload_inner_offset=PAYLOAD_INNER_OFFSET,
            box_size=BOX_SIZE,
            dt=DT
        )
        print(f"Data saved to {data_file}")

    #####################################################
    # CREATE ANIMATION                                  #
    #####################################################

    # Determine output filename
    if OUTPUT_FILENAME is None:
        T = int(time.time())
        output_file = f'./visualizations/hollow_payload_animation_T_{T}.mp4'
    else:
        output_file = OUTPUT_FILENAME

    # Create animation
    create_hollow_payload_animation(
        positions, orientations, velocities, payload_positions, params,
        curvity_values, output_file
    )

    print("\nHollow payload simulation and animation completed successfully!")
