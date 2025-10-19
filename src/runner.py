import numpy as np
import time

from .simulation import simulate_single_step, compute_curvity_from_polarity


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

    # Extract goal parameters
    goal_position = params['goal_position']
    particle_view_range = params['particle_view_range']
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

    # Initialize scalar scores and polarity vectors for all particles
    particle_scores = np.full(n_particles, 9999, dtype=np.int64)
    polarity = np.zeros((n_particles, 2), dtype=np.float64)
    # Initialize all polarity vectors as unit vectors in direction π/4
    angle = np.pi / 4
    polarity[:, 0] = np.cos(angle)
    polarity[:, 1] = np.sin(angle)

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
    saved_polarity = np.zeros((n_saves, n_particles, 2))
    saved_particle_scores = np.zeros((n_saves, n_particles), dtype=np.int64)

    # Compute initial curvity from polarity vectors
    initial_curvity = compute_curvity_from_polarity(orientations, polarity, n_particles)

    # Store initial state
    saved_positions[0] = positions.copy()
    saved_orientations[0] = orientations.copy()
    saved_velocities[0] = velocities.copy()
    saved_payload_positions[0] = payload_pos.copy()
    saved_payload_velocities[0] = payload_vel.copy()
    saved_curvity[0] = initial_curvity.copy()
    saved_polarity[0] = polarity.copy()
    saved_particle_scores[0] = particle_scores.copy()

    # Run simulation
    start_time = time.time()
    save_idx = 1
    goal_reached = False

    for step in range(1, n_steps + 1):
        # Unified simulation step
        positions, orientations, velocities, payload_pos, payload_vel, curvity = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            params['particle_radius'], params['v0'], params['mobility'], params['payload_mobility'],
            polarity, particle_scores, params['stiffness'],
            params['box_size'], params['payload_radius'], params['dt'], params['rot_diffusion'],
            n_particles, step, goal_position, particle_view_range, score_and_polarity_update_interval, walls,
            params['directedness']
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
            saved_polarity[save_idx] = polarity.copy()
            saved_particle_scores[save_idx] = particle_scores.copy()
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
        saved_polarity,
        saved_particle_scores,
        particle_scores,
        polarity,
        end_time - start_time
    )


def save_simulation_data(filename, positions, orientations, velocities, payload_positions,
                        payload_velocities, params, curvity_values, polarity, particle_scores):
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
        polarity=polarity, # Polarity vectors over time
        particle_scores=particle_scores, # Particle scores over time
        # Parameters
        # params['curvity'] accessible through curvity_values[-1]
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
        particle_view_range=params['particle_view_range'],
        score_and_polarity_update_interval=params['score_and_polarity_update_interval'],
        directedness=params['directedness'],
        # Wall parameters
        walls=params['walls']
    )

def extract_simulation_data(filename):
    """Extract simulation data from a file."""
    data = np.load(filename)
    return data
