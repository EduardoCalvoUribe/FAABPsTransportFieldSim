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

    # Extract goal parameters (used only for tracking when payload reaches goal)
    goal_position = params['goal_position']
    score_and_polarity_update_interval = params['score_and_polarity_update_interval']

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

    # Initialize polarity vector fields for all particles
    # Each particle has an NxN matrix of polarity vectors (vector field)
    # Shape: (n_particles, box_size, box_size, 2)
    polarity_fields = np.zeros((n_particles, box_size, box_size, 2), dtype=np.float64)

    # Initialize collision tracking (-1 means no recent collision)
    last_collision_partner = np.full(n_particles, -1, dtype=np.int64)

    # Initialize payload location
    # If circular motion is enabled, start at the beginning of the circle (angle = 0)
    if params.get('payload_circular_motion', False):
        circle_center = params.get('payload_circle_center', np.array([box_size/2, box_size/2]))
        circle_radius = params.get('payload_circle_radius', box_size/3)
        # Start at angle = 0 (rightmost point of circle)
        payload_pos = np.array([
            circle_center[0] + circle_radius,
            circle_center[1]
        ])
    else:
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
    # Store active polarity vectors (fetched from field) for visualization
    saved_active_polarity = np.zeros((n_saves, n_particles, 2))

    # Set initial curvity
    initial_curvity = np.full(n_particles, -1.0)

    # Store initial state
    saved_positions[0] = positions.copy()
    saved_orientations[0] = orientations.copy()
    saved_velocities[0] = velocities.copy()
    saved_payload_positions[0] = payload_pos.copy()
    saved_payload_velocities[0] = payload_vel.copy()
    saved_curvity[0] = initial_curvity.copy()
    saved_active_polarity[0] = np.zeros((n_particles, 2))  # Initially all zero

    # Run simulation
    start_time = time.time()
    save_idx = 1
    goal_reached = False

    for step in range(1, n_steps + 1):
        # Unified simulation step
        positions, orientations, velocities, payload_pos, payload_vel, curvity, active_polarity, last_collision_partner = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            params['particle_radius'], params['v0'], params['mobility'], params['payload_mobility'],
            polarity_fields, params['stiffness'],
            params['box_size'], params['payload_radius'], params['dt'], params['rot_diffusion'],
            n_particles, step, score_and_polarity_update_interval, walls, params.get('n_training_steps', n_steps // 2),
            params.get('payload_circular_motion', False),
            params.get('payload_circle_center', np.array([box_size/2, box_size/2])),
            params.get('payload_circle_radius', box_size/3),
            params.get('payload_n_rotations', 1),
            last_collision_partner,
            params.get('collision_share_interval', 1),
            params.get('polarity_force_scaling', 1.0),
            params.get('test_phase_learning', 2)
        )

        # Check if payload reached goal
        if not goal_reached:
            distance_to_goal = np.sqrt(np.sum((payload_pos - goal_position)**2))
            if distance_to_goal <= params['payload_radius']:
                print(f"Goal reached at step {step}! Distance: {distance_to_goal:.3f}")
                goal_reached = True
                # End simulation early if configured to do so
                if params['end_when_goal_reached']:
                    print("Ending simulation early (END_WHEN_GOAL_REACHED = True)")
                    break

        # Save data at specified intervals
        if step % save_interval == 0:
            saved_positions[save_idx] = positions
            saved_orientations[save_idx] = orientations
            saved_velocities[save_idx] = velocities
            saved_payload_positions[save_idx] = payload_pos
            saved_payload_velocities[save_idx] = payload_vel
            saved_curvity[save_idx] = curvity.copy()
            saved_active_polarity[save_idx] = active_polarity.copy()
            save_idx += 1

            # Report progress periodically
            if step % (save_interval * 10) == 0:
                print(f"Step {step}:")
                payload_displacement = np.sqrt(np.sum((saved_payload_positions[save_idx-1] - saved_payload_positions[0])**2))
                print(f"  Payload position: {payload_pos}")
                print(f"  Payload displacement from start: {payload_displacement:.3f}")
                distance_to_goal = np.sqrt(np.sum((payload_pos - goal_position)**2))
                print(f"  Distance to goal: {distance_to_goal:.3f}")

    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")

    # Trim arrays to only include saved frames
    saved_positions = saved_positions[:save_idx]
    saved_orientations = saved_orientations[:save_idx]
    saved_velocities = saved_velocities[:save_idx]
    saved_payload_positions = saved_payload_positions[:save_idx]
    saved_payload_velocities = saved_payload_velocities[:save_idx]
    saved_curvity = saved_curvity[:save_idx]
    saved_active_polarity = saved_active_polarity[:save_idx]

    # Calculate payload displacement
    total_payload_displacement = np.sqrt(np.sum((saved_payload_positions[-1] - saved_payload_positions[0])**2))
    print(f"Total payload displacement: {total_payload_displacement:.3f}")

    final_distance_to_goal = np.sqrt(np.sum((saved_payload_positions[-1] - goal_position)**2))
    print(f"Final distance to goal: {final_distance_to_goal:.3f}")

    return (
        saved_positions,
        saved_orientations,
        saved_velocities,
        saved_payload_positions,
        saved_payload_velocities,
        saved_curvity,
        saved_active_polarity,
        polarity_fields,
        end_time - start_time
    )


def save_simulation_data(filename, positions, orientations, velocities, payload_positions,
                        payload_velocities, params, curvity_values, active_polarity):
    """Save simulation data including individual particle parameters."""
    np.savez(
        filename,
        # Frame-specific data
        positions=positions,
        orientations=orientations,
        velocities=velocities,
        payload_positions=payload_positions,
        payload_velocities=payload_velocities,
        curvity_values=curvity_values, # Curvity values over time, for each particle
        active_polarity=active_polarity, # Active polarity vectors over time (fetched from field)
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
        # Goal parameters
        goal_position=params['goal_position'],
        force_update_interval=params['score_and_polarity_update_interval'],
        # Wall parameters
        walls=params['walls']
    )

def extract_simulation_data(filename):
    """Extract simulation data from a file."""
    data = np.load(filename)
    return data
