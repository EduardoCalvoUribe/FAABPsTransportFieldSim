import io
import zipfile
import numpy as np
import time

from .simulation import simulate_single_step
from .forces import build_wall_spatial_index


#####################################################
# Main simulation runner functions                  #
#####################################################

def run_payload_simulation(params, light=False):
    """Run the complete payload transport simulation.

    light=True: skip pre-allocating per-particle time-series arrays (positions,
    orientations, velocities, curvity, polarity, scores).  Only payload positions
    are accumulated; the final particle state is returned as a 1-frame array.
    Use this when only the last frame is needed (e.g. render_final_frame.py).
    """
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

    # Build wall spatial index.
    # cell_size: ~box/20 gives a 20×20 grid; fine enough to prune most walls per lookup.
    # margin: set to payload_radius so a single-cell query covers the full force range
    # for both particles (radius ~1) and the payload (radius ~20).
    wall_cell_size = max(box_size / 20.0, 2.0)
    wall_margin = float(params['payload_radius'])
    wall_grid_offsets, wall_grid_indices, n_wall_cells = build_wall_spatial_index(
        walls, box_size, wall_cell_size, wall_margin
    )

    # Extract curvity params
    max_curvity = params['max_curvity']
    min_curvity = params['min_curvity']
    mid_curvity = params['mid_curvity']

    # Extract polarity nudge params
    polarity_nudge_interval = params['polarity_nudge_interval']
    polarity_nudge_strength = params['polarity_nudge_strength']

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
    saved_payload_positions = np.zeros((n_saves, 2))
    saved_payload_velocities = np.zeros((n_saves, 2))
    if not light:
        saved_positions = np.zeros((n_saves, n_particles, 2))
        saved_orientations = np.zeros((n_saves, n_particles, 2))
        saved_velocities = np.zeros((n_saves, n_particles, 2))
        saved_curvity = np.zeros((n_saves, n_particles))
        saved_polarity = np.zeros((n_saves, n_particles, 2))
        saved_particle_scores = np.zeros((n_saves, n_particles), dtype=np.int64)

    # Set initial curvity
    initial_curvity = np.full(n_particles, -1.0)

    # Store initial state
    saved_payload_positions[0] = payload_pos.copy()
    saved_payload_velocities[0] = payload_vel.copy()
    if not light:
        saved_positions[0] = positions.copy()
        saved_orientations[0] = orientations.copy()
        saved_velocities[0] = velocities.copy()
        saved_curvity[0] = initial_curvity.copy()
        saved_polarity[0] = polarity.copy()
        saved_particle_scores[0] = particle_scores.copy()

    # Run simulation
    start_time = time.time()
    save_idx = 1
    goal_reached = False
    final_step = n_steps  # Track actual step count when simulation ends

    for step in range(1, n_steps + 1):
        # Unified simulation step
        positions, orientations, velocities, payload_pos, payload_vel, curvity = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            params['particle_radius'], params['v0'], params['mobility'], params['payload_mobility'],
            polarity, particle_scores, params['stiffness'],
            params['box_size'], params['payload_radius'], params['dt'], params['rot_diffusion'],
            n_particles, step, goal_position, particle_view_range, score_and_polarity_update_interval, walls,
            max_curvity, min_curvity, mid_curvity,
            polarity_nudge_interval, polarity_nudge_strength,
            wall_grid_offsets, wall_grid_indices, n_wall_cells, wall_cell_size
        )

        # Check if payload reached goal
        if not goal_reached:
            distance_to_goal = np.sqrt(np.sum((payload_pos - goal_position)**2))
            if distance_to_goal <= params['payload_radius']:
                print(f"Goal reached at step {step}! Distance: {distance_to_goal:.3f}")
                goal_reached = True
                final_step = step  # Record actual step when goal was reached
                # End simulation early if configured to do so
                if params['end_when_goal_reached']:
                    print("Ending simulation early (END_WHEN_GOAL_REACHED = True)")
                    break

        # Save data at specified intervals
        if step % save_interval == 0:
            saved_payload_positions[save_idx] = payload_pos
            saved_payload_velocities[save_idx] = payload_vel
            if not light:
                saved_positions[save_idx] = positions
                saved_orientations[save_idx] = orientations
                saved_velocities[save_idx] = velocities
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

    # Trim arrays to only include saved frames
    saved_payload_positions = saved_payload_positions[:save_idx]
    saved_payload_velocities = saved_payload_velocities[:save_idx]
    if not light:
        saved_positions = saved_positions[:save_idx]
        saved_orientations = saved_orientations[:save_idx]
        saved_velocities = saved_velocities[:save_idx]
        saved_curvity = saved_curvity[:save_idx]
        saved_polarity = saved_polarity[:save_idx]
        saved_particle_scores = saved_particle_scores[:save_idx]
    else:
        saved_positions = positions[np.newaxis]           # (1, N, 2)
        saved_orientations = orientations[np.newaxis]     # (1, N, 2)
        saved_velocities = velocities[np.newaxis]         # (1, N, 2)
        saved_curvity = curvity[np.newaxis]               # (1, N)
        saved_polarity = polarity[np.newaxis]             # (1, N, 2)
        saved_particle_scores = particle_scores[np.newaxis]  # (1, N)

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
        end_time - start_time,
        final_step  # Actual step count when simulation ended
    )


def run_payload_simulation_from_state(params, initial_positions, initial_orientations,
                                      initial_payload_pos, initial_particle_scores,
                                      initial_polarity, light=True):
    """Continue a simulation from a provided initial state.

    Identical to run_payload_simulation except particle positions, orientations,
    payload position, scores, and polarity are supplied rather than randomised.
    Use this to resume from the last frame of a light NPZ.

    The step counter restarts at 1 so score/polarity updates and polarity nudges
    fire normally from the start of the continuation.
    """
    print(f"Running continuation: {params['n_particles']} particles, {params['n_steps']} steps...")

    n_particles = params['n_particles']
    box_size = params['box_size']
    n_steps = params['n_steps']
    save_interval = params['save_interval']

    goal_position = params['goal_position']
    particle_view_range = params['particle_view_range']
    score_and_polarity_update_interval = params['score_and_polarity_update_interval']

    walls = params['walls']

    wall_cell_size = max(box_size / 20.0, 2.0)
    wall_margin = float(params['payload_radius'])
    wall_grid_offsets, wall_grid_indices, n_wall_cells = build_wall_spatial_index(
        walls, box_size, wall_cell_size, wall_margin
    )

    max_curvity = params['max_curvity']
    min_curvity = params['min_curvity']
    mid_curvity = params['mid_curvity']

    polarity_nudge_interval = params['polarity_nudge_interval']
    polarity_nudge_strength = params['polarity_nudge_strength']

    positions       = initial_positions.copy()
    orientations    = initial_orientations.copy()
    velocities      = np.zeros((n_particles, 2))
    particle_scores = initial_particle_scores.copy()
    polarity        = initial_polarity.copy()
    payload_pos     = initial_payload_pos.copy()
    payload_vel     = np.zeros(2)

    n_saves = n_steps // save_interval + 1
    saved_payload_positions  = np.zeros((n_saves, 2))
    saved_payload_velocities = np.zeros((n_saves, 2))
    if not light:
        saved_positions      = np.zeros((n_saves, n_particles, 2))
        saved_orientations   = np.zeros((n_saves, n_particles, 2))
        saved_velocities     = np.zeros((n_saves, n_particles, 2))
        saved_curvity        = np.zeros((n_saves, n_particles))
        saved_polarity       = np.zeros((n_saves, n_particles, 2))
        saved_particle_scores = np.zeros((n_saves, n_particles), dtype=np.int64)

    initial_curvity = np.full(n_particles, -1.0)

    saved_payload_positions[0]  = payload_pos.copy()
    saved_payload_velocities[0] = payload_vel.copy()
    if not light:
        saved_positions[0]       = positions.copy()
        saved_orientations[0]    = orientations.copy()
        saved_velocities[0]      = velocities.copy()
        saved_curvity[0]         = initial_curvity.copy()
        saved_polarity[0]        = polarity.copy()
        saved_particle_scores[0] = particle_scores.copy()

    start_time = time.time()
    save_idx   = 1
    goal_reached = False
    final_step   = n_steps

    for step in range(1, n_steps + 1):
        positions, orientations, velocities, payload_pos, payload_vel, curvity = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            params['particle_radius'], params['v0'], params['mobility'], params['payload_mobility'],
            polarity, particle_scores, params['stiffness'],
            params['box_size'], params['payload_radius'], params['dt'], params['rot_diffusion'],
            n_particles, step, goal_position, particle_view_range, score_and_polarity_update_interval, walls,
            max_curvity, min_curvity, mid_curvity,
            polarity_nudge_interval, polarity_nudge_strength,
            wall_grid_offsets, wall_grid_indices, n_wall_cells, wall_cell_size
        )

        if not goal_reached:
            distance_to_goal = np.sqrt(np.sum((payload_pos - goal_position)**2))
            if distance_to_goal <= params['payload_radius']:
                print(f"Goal reached at step {step}! Distance: {distance_to_goal:.3f}")
                goal_reached = True
                final_step = step
                if params['end_when_goal_reached']:
                    print("Ending simulation early (END_WHEN_GOAL_REACHED = True)")
                    break

        if step % save_interval == 0:
            saved_payload_positions[save_idx]  = payload_pos
            saved_payload_velocities[save_idx] = payload_vel
            if not light:
                saved_positions[save_idx]       = positions
                saved_orientations[save_idx]    = orientations
                saved_velocities[save_idx]      = velocities
                saved_curvity[save_idx]         = curvity.copy()
                saved_polarity[save_idx]        = polarity.copy()
                saved_particle_scores[save_idx] = particle_scores.copy()
            save_idx += 1

            if step % (save_interval * 10) == 0:
                print(f"Step {step}:")
                payload_displacement = np.sqrt(np.sum((saved_payload_positions[save_idx-1] - saved_payload_positions[0])**2))
                print(f"  Payload position: {payload_pos}")
                print(f"  Payload displacement from continuation start: {payload_displacement:.3f}")
                distance_to_goal = np.sqrt(np.sum((payload_pos - goal_position)**2))
                print(f"  Distance to goal: {distance_to_goal:.3f}")

    end_time = time.time()
    print(f"Continuation completed in {end_time - start_time:.2f} seconds")

    saved_payload_positions  = saved_payload_positions[:save_idx]
    saved_payload_velocities = saved_payload_velocities[:save_idx]
    if not light:
        saved_positions       = saved_positions[:save_idx]
        saved_orientations    = saved_orientations[:save_idx]
        saved_velocities      = saved_velocities[:save_idx]
        saved_curvity         = saved_curvity[:save_idx]
        saved_polarity        = saved_polarity[:save_idx]
        saved_particle_scores = saved_particle_scores[:save_idx]
    else:
        saved_positions       = positions[np.newaxis]        # (1, N, 2)
        saved_orientations    = orientations[np.newaxis]     # (1, N, 2)
        saved_velocities      = velocities[np.newaxis]       # (1, N, 2)
        saved_curvity         = curvity[np.newaxis]          # (1, N)
        saved_polarity        = polarity[np.newaxis]         # (1, N, 2)
        saved_particle_scores = particle_scores[np.newaxis]  # (1, N)

    total_displacement = np.sqrt(np.sum((saved_payload_positions[-1] - saved_payload_positions[0])**2))
    print(f"Continuation payload displacement: {total_displacement:.3f}")

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
        end_time - start_time,
        final_step,
    )


def save_light_simulation_data(filename, positions, payload_positions, curvity_values,
                               particle_scores, params):
    """Save only the data required by render_final_frame.py.

    Keeps only the last frame of per-particle arrays (positions, curvity, scores)
    while retaining the full payload trajectory for the path plot.  The arrays are
    stored as 1-element slices so that data['positions'][-1] etc. still work.
    """
    np.savez(
        filename,
        positions=positions[-1:],             # (1, N, 2)
        payload_positions=payload_positions,   # (T, 2) — full trajectory
        curvity_values=curvity_values[-1:],   # (1, N)
        particle_scores=particle_scores[-1:], # (1, N)
        box_size=params['box_size'],
        payload_radius=params['payload_radius'],
        particle_radius=params['particle_radius'],
        goal_position=params['goal_position'],
        walls=params['walls'],
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
        # Wall parameters
        walls=params['walls']
    )

def extract_simulation_data(filename):
    """Extract simulation data from a file."""
    data = np.load(filename)
    return data


FRAME_ARRAYS = {'positions', 'orientations', 'velocities',
                'payload_positions', 'payload_velocities',
                'curvity_values', 'polarity', 'particle_scores'}

def thin_npz(source_file, dest_file, keep_every=4):
    """Copy source_file to dest_file keeping every keep_every-th frame.

    Processes one array at a time so peak memory is a single array, not the
    entire file.  Works on both compressed and uncompressed NPZ files.
    """
    dest_path = dest_file if dest_file.endswith('.npz') else dest_file + '.npz'
    with zipfile.ZipFile(source_file, 'r') as src_zip, \
         zipfile.ZipFile(dest_path, 'w', compression=zipfile.ZIP_STORED) as dst_zip:
        for name in src_zip.namelist():
            key = name[:-4] if name.endswith('.npy') else name
            with src_zip.open(name) as f:
                arr = np.load(io.BytesIO(f.read()))
            if key in FRAME_ARRAYS and arr.ndim >= 1:
                arr = arr[::keep_every]
            buf = io.BytesIO()
            np.save(buf, arr)
            dst_zip.writestr(name, buf.getvalue())
            del arr, buf
            print(f"  {key}")
    print(f"Thinned file saved to: {dest_path}")
