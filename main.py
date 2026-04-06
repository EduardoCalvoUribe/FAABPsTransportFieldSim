import numpy as np
import time
import os

from src import wilson
from src.runner import run_payload_simulation
from src.visualization import create_payload_animation


def K_to_c(K, x1, y1, x2, y2):
    """Convert standard curvature K = 1/R to the chord-normalised curvature c = chord/(2R).

    Args:
        K: signed curvature (positive = bulges left of p1→p2, negative = right)
        x1, y1, x2, y2: wall endpoints (needed to compute chord length)

    Returns:
        c: chord-normalised curvature; pass as the 5th column of a wall row.
    """
    chord = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return K * chord / 2.0


def maze_to_walls(passages, grid_size, box_size, include_boundary=True):
    cell = box_size / grid_size
    R = cell / 2
    C_ARC = 0.707107  # sin(45°) = sqrt(2)/2 → quarter circle
    walls = []
    if include_boundary:
        walls += [
            [0,        R-5,        0,        box_size-(R)+5, 0],
            [cell/2-5,        0,        box_size-(R)+5, 0,        0],
            [box_size-(R)+5, box_size, R-5,        box_size, 0],
            [box_size, box_size-(R)+5, box_size, R,        0],   
        ]
        walls += [
            [0,        R,        R,        0,            -C_ARC],
            [box_size-(R),        0,        box_size, R, -C_ARC],
            [R, box_size, 0, box_size-(R),        -C_ARC],   
            [box_size-(R), box_size, box_size,        box_size-(R), C_ARC],
        ]
    for r in range(grid_size - 1):
        for c in range(grid_size):
            if (r + 1, c) not in passages.get((r, c), set()):
                y = (r + 1) * cell
                walls.append([c * cell, y, (c + 1) * cell, y, 0])
    for r in range(grid_size):
        for c in range(grid_size - 1):
            if (r, c + 1) not in passages.get((r, c), set()):
                x = (c + 1) * cell
                walls.append([x, r * cell, x, (r + 1) * cell, 0])

    # Add quarter-circle arcs at every interior grid-point corner.
    # At each point (cx, ry) we check which of the four wall segments exist and
    # add an arc for every (horizontal, vertical) pair that meets there.
    # Convention: p1 = horizontal endpoint, p2 = vertical endpoint.
    # Sign rule: c = +C_ARC when dx_H and dy_V point into the same diagonal
    #            (right+up or left+down), -C_ARC for the opposite diagonals.
    for r_int in range(1, grid_size):
        for c_int in range(1, grid_size):
            cx = c_int * cell
            ry = r_int * cell
            h_right = (r_int, c_int)   not in passages.get((r_int - 1, c_int),     set())
            h_left  = (r_int, c_int-1) not in passages.get((r_int - 1, c_int - 1), set())
            v_up    = (r_int, c_int)   not in passages.get((r_int,     c_int - 1), set())
            v_down  = (r_int-1, c_int) not in passages.get((r_int - 1, c_int - 1), set())
            if h_right and v_up:   walls.append([cx + R, ry, cx, ry + R, +C_ARC])
            if h_right and v_down: walls.append([cx + R, ry, cx, ry - R, -C_ARC])
            if h_left  and v_up:   walls.append([cx - R, ry, cx, ry + R, -C_ARC])
            if h_left  and v_down: walls.append([cx - R, ry, cx, ry - R, +C_ARC])

    # Corners where interior walls meet the boundary.
    # The boundary always supplies both V directions (left/right edges) or both H
    # directions (top/bottom edges), so two arcs are added per interior wall end.
    if include_boundary:
        # Left boundary (x=0): horizontal wall goes right from (0, r_int*cell)
        for r_int in range(1, grid_size):
            if (r_int, 0) not in passages.get((r_int - 1, 0), set()):
                ry = r_int * cell
                walls.append([R, ry, 0, ry + R, +C_ARC])  # H_right + V_up
                walls.append([R, ry, 0, ry - R, -C_ARC])  # H_right + V_down

        # Right boundary (x=box_size): horizontal wall goes left from (box_size, r_int*cell)
        for r_int in range(1, grid_size):
            if (r_int, grid_size - 1) not in passages.get((r_int - 1, grid_size - 1), set()):
                ry = r_int * cell
                walls.append([box_size - R, ry, box_size, ry + R, -C_ARC])  # H_left + V_up
                walls.append([box_size - R, ry, box_size, ry - R, +C_ARC])  # H_left + V_down

        # Bottom boundary (y=0): vertical wall goes up from (c_int*cell, 0)
        for c_int in range(1, grid_size):
            if (0, c_int) not in passages.get((0, c_int - 1), set()):
                cx = c_int * cell
                walls.append([cx + R, 0, cx, R, +C_ARC])  # H_right + V_up
                walls.append([cx - R, 0, cx, R, -C_ARC])  # H_left  + V_up

        # Top boundary (y=box_size): vertical wall goes down from (c_int*cell, box_size)
        for c_int in range(1, grid_size):
            if (grid_size - 1, c_int) not in passages.get((grid_size - 1, c_int - 1), set()):
                cx = c_int * cell
                walls.append([cx + R, box_size, cx, box_size - R, -C_ARC])  # H_right + V_down
                walls.append([cx - R, box_size, cx, box_size - R, +C_ARC])  # H_left  + V_down

    return np.array(walls, dtype=np.float64)


#####################################################
# HYPERPARAMETERS - Configure everything here       #
#####################################################

# Set random seed for reproducibility
RANDOM_SEED = 42

# Simulation parameters
N_PARTICLES = 1000 #600
BOX_SIZE = 600 #300
N_STEPS = 10000
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
PAYLOAD_START_POSITION = np.array([30.0, 30.0])

# Force parameters
STIFFNESS = 25.0

# Goal parameters
GOAL_POSITION = np.array([570.0, 570.0]) # np.array([270.0, 270.0])  # Top-right corner
PARTICLE_VIEW_RANGE = 0.2 * 600 * (1/1.414213)  # Range for goal detection #BOX_SIZE = 300
SCORE_AND_POLARITY_UPDATE_INTERVAL = 20  # How often to update scores & polarity (timesteps)
END_WHEN_GOAL_REACHED = True        # If True, simulation ends when payload reaches goal
POLARITY_NUDGE_INTERVAL = 5        # Every N steps, nudge heading toward polarity
POLARITY_NUDGE_STRENGTH = 0.01       # Angular nudge magnitude (radians)

# Wall configuration — generated from a Wilson maze
MAZE_GRID_SIZE = 10 #5  # W×W grid; larger = more cells, narrower corridors
WALLS = maze_to_walls(
    wilson.generate(MAZE_GRID_SIZE, seed=42), # seed=42
    MAZE_GRID_SIZE,
    BOX_SIZE,
)
# WALLS = None  # uncomment to disable walls entirely
# WALLS = np.array([
#     # Boundary walls                                                    c
#     [0, 0, 0, BOX_SIZE,                                                 0],
#     [0, 0, BOX_SIZE, 0,                                                 0],
#     [BOX_SIZE, BOX_SIZE, 0, BOX_SIZE,                                   0],
#     [BOX_SIZE, BOX_SIZE, BOX_SIZE, 0,                                   0],
#     # Inverted Y shape walls
#     # [2 * BOX_SIZE/6, BOX_SIZE, 4 * BOX_SIZE/6, BOX_SIZE,             0], #top wall
#     [2.2 * BOX_SIZE/6, 4 * BOX_SIZE/7, 2.2 * BOX_SIZE/6, BOX_SIZE,    0], #top left
#     [3.8 * BOX_SIZE/6, 4 * BOX_SIZE/7, 3.8 * BOX_SIZE/6, BOX_SIZE,    0], #top right
#     [2.2 * BOX_SIZE/6, 4 * BOX_SIZE/7, 0, 4 * BOX_SIZE/7,             0], # left shoulder
#     [3.8 * BOX_SIZE/6, 4 * BOX_SIZE/7, BOX_SIZE, 4 * BOX_SIZE/7,      0], # right shoulder
#     # [0, 4 * BOX_SIZE/7, 0, 0,                                         0], #bot left
#     # [BOX_SIZE, 4*BOX_SIZE/7, BOX_SIZE, 0,                             0], #bot right
#     # [0, 0, BOX_SIZE, 0,                                                0], #bot
#     [2 * BOX_SIZE/7, 2.5 * BOX_SIZE/7, 2 * BOX_SIZE/7, 0,             0], #inner left
#     [2 * BOX_SIZE/7, 2.5 * BOX_SIZE/7, 5 * BOX_SIZE/7, 2.5*BOX_SIZE/7, 0], #inner top
#     [5 * BOX_SIZE/7, 2.5 * BOX_SIZE/7, 5 * BOX_SIZE/7, 0,             0], #inner right
# ], dtype=np.float64)
# WALLS = np.array([
#     # Boundary walls                                                    c
#     [0, 0, 0, BOX_SIZE,                                                 0],
#     [0, 0, BOX_SIZE, 0,                                                 0],
#     [BOX_SIZE, BOX_SIZE, 0, BOX_SIZE,                                   0],
#     [BOX_SIZE, BOX_SIZE, BOX_SIZE, 0,                                   0],
#     # walls inside
#     # [0, 33.3, 66.6, 33.3,                                             0],
#     # [33.3, 66.6, 100.0, 66.3,                                         0],
# ], dtype=np.float64)


# Visualization parameters
SHOW_VECTORS = True              # Display v vectors as arrows
COLOR_BY_SCORE = False           # If True: color by score, if False: color by curvity
OUTPUT_FILENAME = "D:/PostThesis/visualizations/localoptim0.mp4"           # If None, uses timestamp. Otherwise specify path.
# OUTPUT_FILENAME = "c:/Users/educa/Downloads/test2.mp4"
# Data saving (set to True to save simulation data)
SAVE_DATA = False
DATA_OUTPUT_PATH = "D:/PostThesis/data/snelltest0_fake.npz"                    # If None, uses timestamp. Otherwise specify path.


#####################
# Main execution    #
#####################

# if __name__ == "__main__": # load and render
#     data = np.load("data/snelltesthuge4_5kr60.npz")
#     saved_positions = data['positions']
#     saved_orientations = data['orientations']
#     saved_velocities = data['velocities']
#     saved_payload_positions = data['payload_positions']
#     saved_curvity = data['curvity_values']
#     saved_polarity = data['polarity']
#     saved_particle_scores = data['particle_scores']
#     params = {
#         'n_particles': saved_positions.shape[1],
#         'box_size': float(data['box_size']),
#         'payload_radius': float(data['payload_radius']),
#         'goal_position': data['goal_position'],
#         'particle_radius': data['particle_radius'],
#         'rot_diffusion': data['rot_diffusion'],
#         'mobility': data['mobility'],
#         'payload_mobility': float(data['payload_mobility']),
#         'walls': data['walls'],
#     }
#     create_payload_animation(
#         saved_positions, saved_orientations, saved_velocities,
#         saved_payload_positions, params, saved_curvity,
#         output_file=OUTPUT_FILENAME,
#         show_vectors=SHOW_VECTORS,
#         polarity=saved_polarity,
#         particle_scores=saved_particle_scores if COLOR_BY_SCORE else None
#     )
        
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
