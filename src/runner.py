import numpy as np
import time

from .simulation import (simulate_single_step, simulate_single_step_hollow_payload,
                         simulate_single_step_multi_hollow_payload)


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
    positions = np.random.uniform((box_size/2)-5, (box_size/2)+5, (n_particles, 2))
    orientations = np.zeros((n_particles, 2))
    velocities = np.zeros((n_particles, 2))

    # Initialize random orientations
    for i in range(n_particles):
        angle = np.random.uniform(0, 2*np.pi)
        orientations[i] = np.array([np.cos(angle), np.sin(angle)])

    # Initialize payload location from parameters
    payload_pos = params['payload_position'].copy()
    payload_vel = np.zeros(2)

    # Pre-allocate arrays for storing simulation data
    n_saves = n_steps // save_interval + 1
    saved_positions = np.zeros((n_saves, n_particles, 2))
    saved_orientations = np.zeros((n_saves, n_particles, 2))
    saved_velocities = np.zeros((n_saves, n_particles, 2))
    saved_payload_positions = np.zeros((n_saves, 2))
    saved_payload_velocities = np.zeros((n_saves, 2))
    saved_curvity = np.zeros((n_saves, n_particles))

    # Get fixed curvity from params
    curvity = params['curvity'].copy()

    # Store initial state
    saved_positions[0] = positions.copy()
    saved_orientations[0] = orientations.copy()
    saved_velocities[0] = velocities.copy()
    saved_payload_positions[0] = payload_pos.copy()
    saved_payload_velocities[0] = payload_vel.copy()
    saved_curvity[0] = curvity.copy()

    # Run simulation
    start_time = time.time()
    save_idx = 1

    for step in range(1, n_steps + 1):
        # Unified simulation step
        positions, orientations, velocities, payload_pos, payload_vel = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            params['particle_radius'], params['v0'], params['mobility'], params['payload_mobility'],
            curvity, params['stiffness'],
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
        saved_curvity,
        end_time - start_time
    )


def run_hollow_payload_simulation(params):
    """Run the complete hollow payload transport simulation.

    The hollow payload is a single boundary circle.
    Particles can push from inside (outward) or outside (inward).

    Optional param 'n_particles_inside': number of particles to initialize inside the hollow.
    """
    print(f"Running hollow payload transport simulation with {params['n_particles']} particles for {params['n_steps']} steps...")

    # Initialize arrays
    n_particles = params['n_particles']
    box_size = params['box_size']
    n_steps = params['n_steps']
    save_interval = params['save_interval']

    # Extract walls
    walls = params['walls']

    # Initialize payload location from parameters (needed for particle placement)
    payload_pos = params['payload_position'].copy()
    payload_radius = params['payload_radius']
    payload_vel = np.zeros(2)

    positions = np.zeros((n_particles, 2))
    orientations = np.zeros((n_particles, 2))
    velocities = np.zeros((n_particles, 2))

    # Place particles inside and outside the hollow payload
    n_inside = params.get('n_particles_inside', 0)
    particle_r = params['particle_radius'][0] if hasattr(params['particle_radius'], '__len__') else params['particle_radius']

    # Place first n_inside particles inside the hollow payload
    if n_inside > 0:
        print(f"Placing {n_inside} particles inside the hollow payload...")
        max_r = payload_radius - particle_r - 0.5
        if max_r > 0:
            for i in range(min(n_inside, n_particles)):
                r = np.sqrt(np.random.uniform(0, 1)) * max_r
                theta = np.random.uniform(0, 2 * np.pi)
                positions[i] = payload_pos + np.array([r * np.cos(theta), r * np.sin(theta)])
        else:
            print("Warning: Hollow payload too small for particles, skipping inside placement")

    # Place remaining particles outside the payload (rejection sampling)
    min_dist = payload_radius + particle_r + 0.5
    for i in range(n_inside, n_particles):
        while True:
            pos = np.random.uniform(0, box_size, 2)
            dist = np.sqrt((pos[0] - payload_pos[0])**2 + (pos[1] - payload_pos[1])**2)
            if dist > min_dist:
                positions[i] = pos
                break

    # Initialize random orientations
    for i in range(n_particles):
        angle = np.random.uniform(0, 2*np.pi)
        orientations[i] = np.array([np.cos(angle), np.sin(angle)])

    # Pre-allocate arrays for storing simulation data
    n_saves = n_steps // save_interval + 1
    saved_positions = np.zeros((n_saves, n_particles, 2))
    saved_orientations = np.zeros((n_saves, n_particles, 2))
    saved_velocities = np.zeros((n_saves, n_particles, 2))
    saved_payload_positions = np.zeros((n_saves, 2))
    saved_payload_velocities = np.zeros((n_saves, 2))
    saved_curvity = np.zeros((n_saves, n_particles))

    # Get fixed curvity from params
    curvity = params['curvity'].copy()

    # Store initial state
    saved_positions[0] = positions.copy()
    saved_orientations[0] = orientations.copy()
    saved_velocities[0] = velocities.copy()
    saved_payload_positions[0] = payload_pos.copy()
    saved_payload_velocities[0] = payload_vel.copy()
    saved_curvity[0] = curvity.copy()

    # Run simulation
    start_time = time.time()
    save_idx = 1

    for step in range(1, n_steps + 1):
        # Hollow payload simulation step
        positions, orientations, velocities, payload_pos, payload_vel = simulate_single_step_hollow_payload(
            positions, orientations, velocities, payload_pos, payload_vel,
            params['particle_radius'], params['v0'], params['mobility'], params['payload_mobility'],
            curvity, params['stiffness'],
            params['box_size'], params['payload_radius'],
            params['dt'], params['rot_diffusion'],
            n_particles, walls
        )

        # Save data at specified intervals
        if step % save_interval == 0:
            saved_positions[save_idx] = positions
            saved_orientations[save_idx] = orientations
            saved_velocities[save_idx] = velocities
            saved_payload_positions[save_idx] = payload_pos
            saved_payload_velocities[save_idx] = payload_vel
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
        saved_curvity,
        end_time - start_time
    )


def run_multi_hollow_payload_simulation(params):
    """Run simulation with multiple hollow payloads.

    10 small hollow payloads inside 1 large enclosing hollow payload.
    Particles are distributed evenly inside the small payloads.
    """
    n_particles = params['n_particles']
    n_small_payloads = params['n_small_payloads']
    box_size = params['box_size']
    n_steps = params['n_steps']
    save_interval = params['save_interval']
    walls = params['walls']

    print(f"Running multi-hollow payload simulation:")
    print(f"  {n_particles} particles, {n_small_payloads} small payloads, {n_steps} steps")

    # Initialize small payload positions and velocities
    small_payload_positions = params['small_payload_positions'].copy()
    small_payload_velocities = np.zeros((n_small_payloads, 2))
    small_payload_radii = params['small_payload_radii']
    small_payload_mobilities = params['small_payload_mobilities']

    # Initialize large payload
    large_payload_pos = params['large_payload_position'].copy()
    large_payload_vel = np.zeros(2)
    large_payload_radius = params['large_payload_radius']
    large_payload_mobility = params['large_payload_mobility']

    # Initialize particle arrays
    positions = np.zeros((n_particles, 2))
    orientations = np.zeros((n_particles, 2))
    velocities = np.zeros((n_particles, 2))

    # Get particle radius
    particle_r = params['particle_radius'][0] if hasattr(params['particle_radius'], '__len__') else params['particle_radius']

    # Distribute particles evenly among small payloads (inside each)
    particles_per_payload = n_particles // n_small_payloads
    particle_idx = 0

    for p_idx in range(n_small_payloads):
        payload_center = small_payload_positions[p_idx]
        payload_r = small_payload_radii[p_idx]
        max_r = payload_r - particle_r - 0.5  # Leave margin

        # Last payload gets any remaining particles
        n_to_place = particles_per_payload
        if p_idx == n_small_payloads - 1:
            n_to_place = n_particles - particle_idx

        for _ in range(n_to_place):
            if max_r > 0:
                # Random position inside hollow payload
                r = np.sqrt(np.random.uniform(0, 1)) * max_r
                theta = np.random.uniform(0, 2 * np.pi)
                positions[particle_idx] = payload_center + np.array([
                    r * np.cos(theta), r * np.sin(theta)
                ])
            else:
                positions[particle_idx] = payload_center.copy()
            particle_idx += 1

    # Initialize random orientations
    for i in range(n_particles):
        angle = np.random.uniform(0, 2 * np.pi)
        orientations[i] = np.array([np.cos(angle), np.sin(angle)])

    # Pre-allocate saved arrays
    n_saves = n_steps // save_interval + 1
    saved_positions = np.zeros((n_saves, n_particles, 2))
    saved_orientations = np.zeros((n_saves, n_particles, 2))
    saved_velocities = np.zeros((n_saves, n_particles, 2))
    saved_small_payload_positions = np.zeros((n_saves, n_small_payloads, 2))
    saved_small_payload_velocities = np.zeros((n_saves, n_small_payloads, 2))
    saved_large_payload_positions = np.zeros((n_saves, 2))
    saved_large_payload_velocities = np.zeros((n_saves, 2))
    saved_curvity = np.zeros((n_saves, n_particles))

    curvity = params['curvity'].copy()

    # Store initial state
    saved_positions[0] = positions.copy()
    saved_orientations[0] = orientations.copy()
    saved_velocities[0] = velocities.copy()
    saved_small_payload_positions[0] = small_payload_positions.copy()
    saved_small_payload_velocities[0] = small_payload_velocities.copy()
    saved_large_payload_positions[0] = large_payload_pos.copy()
    saved_large_payload_velocities[0] = large_payload_vel.copy()
    saved_curvity[0] = curvity.copy()

    # Main simulation loop
    start_time = time.time()
    save_idx = 1

    for step in range(1, n_steps + 1):
        (positions, orientations, velocities,
         small_payload_positions, small_payload_velocities,
         large_payload_pos, large_payload_vel) = simulate_single_step_multi_hollow_payload(
            positions, orientations, velocities,
            small_payload_positions, small_payload_velocities,
            large_payload_pos, large_payload_vel,
            params['particle_radius'], params['v0'], params['mobility'],
            small_payload_mobilities, large_payload_mobility,
            curvity, params['stiffness'], box_size,
            small_payload_radii, large_payload_radius,
            params['dt'], params['rot_diffusion'],
            n_particles, n_small_payloads, walls
        )

        if step % save_interval == 0:
            saved_positions[save_idx] = positions
            saved_orientations[save_idx] = orientations
            saved_velocities[save_idx] = velocities
            saved_small_payload_positions[save_idx] = small_payload_positions
            saved_small_payload_velocities[save_idx] = small_payload_velocities
            saved_large_payload_positions[save_idx] = large_payload_pos
            saved_large_payload_velocities[save_idx] = large_payload_vel
            saved_curvity[save_idx] = curvity.copy()
            save_idx += 1

            if step % (save_interval * 10) == 0:
                print(f"Step {step}")

    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")

    # Trim arrays
    saved_positions = saved_positions[:save_idx]
    saved_orientations = saved_orientations[:save_idx]
    saved_velocities = saved_velocities[:save_idx]
    saved_small_payload_positions = saved_small_payload_positions[:save_idx]
    saved_small_payload_velocities = saved_small_payload_velocities[:save_idx]
    saved_large_payload_positions = saved_large_payload_positions[:save_idx]
    saved_large_payload_velocities = saved_large_payload_velocities[:save_idx]
    saved_curvity = saved_curvity[:save_idx]

    return (
        saved_positions,
        saved_orientations,
        saved_velocities,
        saved_small_payload_positions,
        saved_small_payload_velocities,
        saved_large_payload_positions,
        saved_large_payload_velocities,
        saved_curvity,
        end_time - start_time
    )


def save_simulation_data(filename, positions, orientations, velocities, payload_positions,
                        payload_velocities, params, curvity_values):
    """Save simulation data including individual particle parameters."""
    np.savez(
        filename,
        # Frame-specific data
        positions=positions,
        orientations=orientations,
        velocities=velocities,
        payload_positions=payload_positions,
        payload_velocities=payload_velocities,
        curvity_values=curvity_values, # Curvity values over time, for each particle (fixed)
        # Parameters
        v0=params['v0'],
        mobility=params['mobility'],
        particle_radius=params['particle_radius'],
        payload_mobility=params['payload_mobility'],
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
