import numpy as np
import time

from .simulation import simulate_single_step


#####################################################
# Main simulation runner functions                  #
#####################################################

def run_payload_simulation(params):
    """Run the complete payload transport simulation."""
    print(f"Running payload transport simulation with {params['n_particles']} particles for {params['n_steps']} steps...")

    # Initialize arrays
    n_particles = params['n_particles']
    box_size = params['box_size']
    n_steps = params['n_steps']
    save_interval = params['save_interval']

    # Extract walls
    walls = params['walls']

    # Initialize particle positions, orientations, and velocities
    positions = np.random.uniform(0, box_size, (n_particles, 2))
    orientations = np.zeros((n_particles, 2))
    velocities = np.zeros((n_particles, 2))

    # Initialize random orientations
    for i in range(n_particles):
        angle = np.random.uniform(0, 2*np.pi)
        orientations[i] = np.array([np.cos(angle), np.sin(angle)])

    # Initialize payload location and orientation from parameters
    payload_pos = params['payload_position'].copy()
    payload_vel = np.zeros(2)
    payload_angle = np.random.uniform(0, 2*np.pi)
    payload_orientation = np.array([np.cos(payload_angle), np.sin(payload_angle)])

    # Pre-allocate arrays for storing simulation data
    n_saves = n_steps // save_interval + 1
    saved_positions = np.zeros((n_saves, n_particles, 2))
    saved_orientations = np.zeros((n_saves, n_particles, 2))
    saved_velocities = np.zeros((n_saves, n_particles, 2))
    saved_payload_positions = np.zeros((n_saves, 2))
    saved_payload_velocities = np.zeros((n_saves, 2))
    saved_payload_orientations = np.zeros((n_saves, 2))
    saved_curvity = np.zeros((n_saves, n_particles))

    # Get fixed curvity from params
    curvity = params['curvity'].copy()

    # Store initial state
    saved_positions[0] = positions.copy()
    saved_orientations[0] = orientations.copy()
    saved_velocities[0] = velocities.copy()
    saved_payload_positions[0] = payload_pos.copy()
    saved_payload_velocities[0] = payload_vel.copy()
    saved_payload_orientations[0] = payload_orientation.copy()
    saved_curvity[0] = curvity.copy()

    # Run simulation
    start_time = time.time()
    save_idx = 1

    for step in range(1, n_steps + 1):
        # Unified simulation step
        positions, orientations, velocities, payload_pos, payload_vel, payload_orientation = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel, payload_orientation,
            params['particle_radius'], params['v0'], params['mobility'], params['payload_mobility'],
            params['payload_v0'], params['payload_rot_diffusion'], curvity, params['stiffness'],
            params['box_size'], params['payload_radius'], params['dt'], params['rot_diffusion'],
            n_particles, walls
        )

        # Save data at specified intervals
        if step % save_interval == 0:
            saved_positions[save_idx] = positions
            saved_orientations[save_idx] = orientations
            saved_velocities[save_idx] = velocities
            saved_payload_positions[save_idx] = payload_pos
            saved_payload_velocities[save_idx] = payload_vel
            saved_payload_orientations[save_idx] = payload_orientation
            saved_curvity[save_idx] = curvity.copy()
            save_idx += 1

            # Report progress periodically
            if step % (save_interval * 10) == 0:
                print(f"Step {step}:")
                payload_displacement = np.sqrt(np.sum((saved_payload_positions[save_idx-1] - saved_payload_positions[0])**2))
                print(f"  Payload position: {payload_pos}")
                print(f"  Payload displacement from start: {payload_displacement:.3f}")

    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")

    # Trim arrays to only include saved frames
    saved_positions = saved_positions[:save_idx]
    saved_orientations = saved_orientations[:save_idx]
    saved_velocities = saved_velocities[:save_idx]
    saved_payload_positions = saved_payload_positions[:save_idx]
    saved_payload_velocities = saved_payload_velocities[:save_idx]
    saved_payload_orientations = saved_payload_orientations[:save_idx]
    saved_curvity = saved_curvity[:save_idx]

    # Calculate payload displacement
    total_payload_displacement = np.sqrt(np.sum((saved_payload_positions[-1] - saved_payload_positions[0])**2))
    print(f"Total payload displacement: {total_payload_displacement:.3f}")

    return (
        saved_positions,
        saved_orientations,
        saved_velocities,
        saved_payload_positions,
        saved_payload_velocities,
        saved_payload_orientations,
        saved_curvity,
        end_time - start_time
    )


def save_simulation_data(filename, positions, orientations, velocities, payload_positions,
                        payload_velocities, payload_orientations, params, curvity_values):
    """Save simulation data including individual particle parameters."""
    np.savez(
        filename,
        # Frame-specific data
        positions=positions,
        orientations=orientations,
        velocities=velocities,
        payload_positions=payload_positions,
        payload_velocities=payload_velocities,
        payload_orientations=payload_orientations,
        curvity_values=curvity_values, # Curvity values over time, for each particle (fixed)
        # Parameters
        v0=params['v0'],
        mobility=params['mobility'],
        particle_radius=params['particle_radius'],
        payload_mobility=params['payload_mobility'],
        payload_v0=params['payload_v0'],
        payload_rot_diffusion=params['payload_rot_diffusion'],
        payload_radius=params['payload_radius'],
        box_size=params['box_size'],
        dt=params['dt'],
        stiffness=params['stiffness'],
        rot_diffusion=params['rot_diffusion'],
        curvity=params['curvity'],  # Fixed curvity values
        # Wall parameters
        walls=params['walls']
    )

def extract_simulation_data(filename):
    """Extract simulation data from a file."""
    data = np.load(filename)
    return data
