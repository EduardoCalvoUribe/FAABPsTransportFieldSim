import numpy as np
import time
import os

from src.runner import run_multi_hollow_payload_simulation
from src.visualization import create_multi_hollow_payload_animation


#####################################################
# HYPERPARAMETERS - Configure everything here       #
#####################################################

# Set random seed for reproducibility
RANDOM_SEED = 42

# Particle distribution by curvity
CURVITY_DISTRIBUTION = {
    0.1: 200,  # 200 particles with curvity 0.1
}

# Total particles = sum of all counts
N_PARTICLES = sum(CURVITY_DISTRIBUTION.values())

BOX_SIZE = 300
N_STEPS = 100000
SAVE_INTERVAL = 10
DT = 0.01

# Particle parameters
PARTICLE_RADIUS = 1.0
PARTICLE_V0 = 3.75              # Self-propulsion speed
PARTICLE_MOBILITY = 1.0
ROTATIONAL_DIFFUSION = 0.05     # Orientational noise

# Small hollow payloads (10 of them)
N_SMALL_PAYLOADS = 10
SMALL_PAYLOAD_RADIUS = 10.0
SMALL_PAYLOAD_MOBILITY = 1 / SMALL_PAYLOAD_RADIUS

# Large enclosing hollow payload
# Must be large enough to contain all small payloads when clustered
LARGE_PAYLOAD_RADIUS = 80.0
LARGE_PAYLOAD_MOBILITY = 1 / LARGE_PAYLOAD_RADIUS

# Force parameters
STIFFNESS = 25.0

# Wall configuration (set to None for no walls)
WALLS = None

# Visualization parameters
OUTPUT_FILENAME = None  # If None, uses timestamp

# Data saving
SAVE_DATA = False
DATA_OUTPUT_PATH = None


#####################################################
# HELPER FUNCTIONS                                  #
#####################################################

def generate_clustered_small_payload_positions(n_payloads, payload_radius, cluster_center, spacing_factor=2.5):
    """
    Generate non-overlapping positions for small payloads in a hexagonal-like cluster.

    Args:
        n_payloads: number of small payloads
        payload_radius: radius of each small payload
        cluster_center: center of the cluster (numpy array)
        spacing_factor: multiplier for payload_radius to determine spacing

    Returns:
        positions: numpy array of shape (n_payloads, 2)
    """
    positions = []
    spacing = spacing_factor * payload_radius  # Minimum spacing between payload centers

    # Place in roughly hexagonal rings
    placed = 0
    ring = 0

    while placed < n_payloads:
        if ring == 0:
            # Center position
            positions.append(cluster_center.copy())
            placed += 1
        else:
            # Place in ring - 6 positions per ring level
            n_in_ring = 6 * ring
            for i in range(n_in_ring):
                if placed >= n_payloads:
                    break
                angle = 2 * np.pi * i / n_in_ring
                r = ring * spacing
                pos = cluster_center + np.array([r * np.cos(angle), r * np.sin(angle)])
                positions.append(pos)
                placed += 1
        ring += 1

    return np.array(positions)


#####################################################
# Main execution                                    #
#####################################################

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

    print("Compiling JIT functions for multi-hollow payload...")

    # Build parameter dictionary for compilation run (small scale)
    compile_n_particles = 10
    compile_n_small_payloads = 3
    compile_center = np.array([BOX_SIZE/2, BOX_SIZE/2])

    compile_small_positions = generate_clustered_small_payload_positions(
        compile_n_small_payloads, SMALL_PAYLOAD_RADIUS, compile_center
    )

    compile_params = {
        'n_particles': compile_n_particles,
        'n_small_payloads': compile_n_small_payloads,
        'box_size': BOX_SIZE,
        'dt': DT,
        'n_steps': 10,
        'save_interval': SAVE_INTERVAL,
        'stiffness': STIFFNESS,
        'walls': WALLS if WALLS is not None else np.zeros((0, 5), dtype=np.float64),

        # Small payloads
        'small_payload_positions': compile_small_positions,
        'small_payload_radii': np.ones(compile_n_small_payloads) * SMALL_PAYLOAD_RADIUS,
        'small_payload_mobilities': np.ones(compile_n_small_payloads) * SMALL_PAYLOAD_MOBILITY,

        # Large payload
        'large_payload_position': compile_center.copy(),
        'large_payload_radius': LARGE_PAYLOAD_RADIUS,
        'large_payload_mobility': LARGE_PAYLOAD_MOBILITY,

        # Particle arrays
        'v0': np.ones(compile_n_particles) * PARTICLE_V0,
        'curvity': np.zeros(compile_n_particles),
        'particle_radius': np.ones(compile_n_particles) * PARTICLE_RADIUS,
        'mobility': np.ones(compile_n_particles) * PARTICLE_MOBILITY,
        'rot_diffusion': np.ones(compile_n_particles) * ROTATIONAL_DIFFUSION,
    }

    run_multi_hollow_payload_simulation(compile_params)
    print("JIT compilation complete.\n")

    #####################################################
    # BUILD SIMULATION PARAMETERS                       #
    #####################################################

    # Build curvity array from distribution dictionary
    curvity_array = []
    for curvity_value, count in CURVITY_DISTRIBUTION.items():
        curvity_array.extend([curvity_value] * count)
    curvity_array = np.array(curvity_array)

    # Generate small payload positions (clustered at center)
    cluster_center = np.array([BOX_SIZE/2, BOX_SIZE/2])
    small_payload_positions = generate_clustered_small_payload_positions(
        N_SMALL_PAYLOADS, SMALL_PAYLOAD_RADIUS, cluster_center
    )

    params = {
        # Global parameters
        'n_particles': N_PARTICLES,
        'n_small_payloads': N_SMALL_PAYLOADS,
        'box_size': BOX_SIZE,
        'dt': DT,
        'n_steps': N_STEPS,
        'save_interval': SAVE_INTERVAL,
        'stiffness': STIFFNESS,

        # Wall parameters
        'walls': WALLS if WALLS is not None else np.zeros((0, 5), dtype=np.float64),

        # Small payloads
        'small_payload_positions': small_payload_positions,
        'small_payload_radii': np.ones(N_SMALL_PAYLOADS) * SMALL_PAYLOAD_RADIUS,
        'small_payload_mobilities': np.ones(N_SMALL_PAYLOADS) * SMALL_PAYLOAD_MOBILITY,

        # Large payload
        'large_payload_position': cluster_center.copy(),
        'large_payload_radius': LARGE_PAYLOAD_RADIUS,
        'large_payload_mobility': LARGE_PAYLOAD_MOBILITY,

        # Particle-specific parameters (arrays)
        'v0': np.ones(N_PARTICLES) * PARTICLE_V0,
        'curvity': curvity_array,
        'particle_radius': np.ones(N_PARTICLES) * PARTICLE_RADIUS,
        'mobility': np.ones(N_PARTICLES) * PARTICLE_MOBILITY,
        'rot_diffusion': np.ones(N_PARTICLES) * ROTATIONAL_DIFFUSION,
    }

    #####################################################
    # RUN SIMULATION                                    #
    #####################################################

    (positions, orientations, velocities,
     small_payload_positions_saved, small_payload_velocities,
     large_payload_positions, large_payload_velocities,
     curvity_values, runtime) = run_multi_hollow_payload_simulation(params)

    #####################################################
    # SAVE DATA (optional)                              #
    #####################################################

    if SAVE_DATA:
        if DATA_OUTPUT_PATH is None:
            T = int(time.time())
            data_file = f'./data/multi_hollow_payload_data_T_{T}.npz'
        else:
            data_file = DATA_OUTPUT_PATH
        np.savez(
            data_file,
            positions=positions,
            orientations=orientations,
            velocities=velocities,
            small_payload_positions=small_payload_positions_saved,
            small_payload_velocities=small_payload_velocities,
            large_payload_positions=large_payload_positions,
            large_payload_velocities=large_payload_velocities,
            curvity_values=curvity_values,
            small_payload_radii=params['small_payload_radii'],
            large_payload_radius=LARGE_PAYLOAD_RADIUS,
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
        output_file = f'./visualizations/multi_hollow_payload_T_{T}.mp4'
    else:
        output_file = OUTPUT_FILENAME

    # Create animation
    create_multi_hollow_payload_animation(
        positions, orientations, velocities,
        small_payload_positions_saved, large_payload_positions,
        params, curvity_values, output_file
    )

    print("\nMulti-hollow payload simulation and animation completed successfully!")
