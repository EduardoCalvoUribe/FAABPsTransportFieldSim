import sys
import os

# Ensure project root is in sys.path regardless of invocation directory,
# and that fig6/ is in sys.path so maze_pruning can be imported directly.
_FIG6_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_FIG6_DIR)
for _p in (_PROJECT_ROOT, _FIG6_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import time

from src import wilson
from src.runner import run_payload_simulation, thin_npz, save_light_simulation_data
from src.visualization import create_payload_animation
from maze_pruning import get_interior_wall_keys, build_pruned_passages


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
RANDOM_SEED = 100

# Simulation parameters
N_PARTICLES = 6000 #1000 #9000 #4000
BOX_SIZE = 1200.0 #600 #1800.0 #1200.0
MAZE_GRID_SIZE = 20 #10 #30 #20
N_STEPS = 100000000

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

# Force parameters
STIFFNESS = 25.0

# Wall configuration — generated from a Wilson maze
WALLS = maze_to_walls(
    wilson.generate(MAZE_GRID_SIZE, seed=42), # seed=42
    MAZE_GRID_SIZE,
    BOX_SIZE,
)
CELL_SIZE = BOX_SIZE / MAZE_GRID_SIZE

# Goal parameters
PAYLOAD_START_POSITION = np.array([CELL_SIZE / 2, CELL_SIZE / 2])
GOAL_POSITION =np.array([BOX_SIZE - (CELL_SIZE / 2), BOX_SIZE - (CELL_SIZE / 2)])# np.array([610.0, 610.0])#  # np.array([270.0, 270.0])  # Top-right corner
PARTICLE_VIEW_RANGE = CELL_SIZE * 1.5 #1200 # CELL_SIZE * 1.5 # Range for goal & neighbor detection
SCORE_AND_POLARITY_UPDATE_INTERVAL = 20  # How often to update scores & polarity (timesteps)
END_WHEN_GOAL_REACHED = True        # If True, simulation ends when payload reaches goal
POLARITY_NUDGE_INTERVAL = 5        # Every N steps, nudge heading toward polarity
POLARITY_NUDGE_STRENGTH = 0.01       # Angular nudge magnitude (radians)


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
#     # [BOX_SIZE * 0.8, BOX_SIZE, BOX_SIZE, BOX_SIZE * 0.8,                0],
# ], dtype=np.float64)


# Visualization parameters

CREATE_VIDEO = False
SHOW_VECTORS = False              # Display polarity vectors as arrows
COLOR_BY_SCORE = False           # If True: color by score, if False: color by curvity
OUTPUT_FILENAME = "D:/PostThesis/visualizations/snell_9000_30_10k.mp4"

# Data saving

SAVE_DATA = True
LIGHT_NPZ = True                                                    # If True, save only data needed by render_final_frame.py (much smaller)
DATA_OUTPUT_PATH = "data/snell_9000_30_2m.npz"                    # If None, uses timestamp. Otherwise specify path.


#####################
# Main execution    #
#####################

# if __name__ == "__main__":  # load and render
#     SOURCE_FILE = "D:/PostThesis/data/snell_4000_20_1m.npz"

#     if OUTPUT_FILENAME:
#         os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)

#     print(f"Loading data from {SOURCE_FILE} ...")
#     t0 = time.time()
#     data = np.load(SOURCE_FILE, mmap_mode='r')
#     saved_positions = data['positions']
#     saved_payload_positions = data['payload_positions']
#     saved_curvity = data['curvity_values']
#     saved_polarity = data['polarity']
#     saved_particle_scores = data['particle_scores']
#     print(f"  positions shape : {saved_positions.shape}")
#     print(f"  payload shape   : {saved_payload_positions.shape}")
#     print(f"Data mapped in {time.time() - t0:.1f}s")
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
#         saved_positions, None, None,
#         saved_payload_positions, params, saved_curvity,
#         output_file=OUTPUT_FILENAME,
#         show_vectors=SHOW_VECTORS,
#         polarity=saved_polarity,
#         particle_scores=saved_particle_scores if COLOR_BY_SCORE else None
#     )

if __name__ == "__main__":  # simulation runner — wall-pruning experiment
    import argparse
    parser = argparse.ArgumentParser(
        description="Wall-pruning experiment: run M+1 simulations, each with one more "
                    "interior wall removed from the maze."
    )
    parser.add_argument("maze_grid_size", type=int,
                        help="Grid size G of the maze (box = 60*G)")
    parser.add_argument("prune_step", type=int,
                        help="How many interior walls to prune before this run "
                             "(0 = full maze, M = all interior walls removed)")
    parser.add_argument("output_name", type=str,
                        help="Base filename stem for data/NPZ output (no extension)")
    parser.add_argument("--n-particles", type=int, default=None,
                        help="Override the default N = 10 * G² particle count")
    parser.add_argument("--prune-seed", type=int, default=7,
                        help="RNG seed governing the wall-pruning order (default: 7)")
    args = parser.parse_args()

    MAZE_GRID_SIZE = args.maze_grid_size
    BOX_SIZE = 60.0 * MAZE_GRID_SIZE
    N_PARTICLES = 10 * MAZE_GRID_SIZE ** 2
    if args.n_particles is not None:
        N_PARTICLES = args.n_particles

    # --- Build pruned maze ---
    original_passages = wilson.generate(MAZE_GRID_SIZE, seed=42)
    all_wall_keys = get_interior_wall_keys(original_passages, MAZE_GRID_SIZE)
    M = len(all_wall_keys)  # (G-1)^2 for a perfect maze

    prune_step = args.prune_step
    if not (0 <= prune_step <= M):
        raise ValueError(f"prune_step={prune_step} is out of range [0, {M}]")

    passages = build_pruned_passages(original_passages, all_wall_keys,
                                     prune_step, args.prune_seed)
    WALLS = maze_to_walls(passages, MAZE_GRID_SIZE, BOX_SIZE)

    walls_remaining = M - prune_step
    print(f"Maze G={MAZE_GRID_SIZE}: M={M} interior walls total, "
          f"prune_step={prune_step}, walls_remaining={walls_remaining}")

    CELL_SIZE = BOX_SIZE / MAZE_GRID_SIZE
    PAYLOAD_START_POSITION = np.array([CELL_SIZE / 2, CELL_SIZE / 2])
    GOAL_POSITION = np.array([BOX_SIZE - (CELL_SIZE / 2), BOX_SIZE - (CELL_SIZE / 2)])
    PARTICLE_VIEW_RANGE = CELL_SIZE * 1.5

    DATA_OUTPUT_PATH = f"data/{args.output_name}.npz"
    OUTPUT_FILENAME = f"visualizations/{args.output_name}.mp4"

    # Set random seed
    np.random.seed(RANDOM_SEED)

    # Create directories if they don't exist
    if SAVE_DATA:
        os.makedirs('./data', exist_ok=True)
    if CREATE_VIDEO:
        os.makedirs('./visualizations', exist_ok=True)

    #####################################################
    # JIT COMPILATION                                   #
    #####################################################

    print("Compiling JIT functions...")

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
        'walls': WALLS,
        'v0': np.ones(compile_n_particles) * PARTICLE_V0,
        'curvity': np.zeros(compile_n_particles),
        'particle_radius': np.ones(compile_n_particles) * PARTICLE_RADIUS,
        'mobility': np.ones(compile_n_particles) * PARTICLE_MOBILITY,
        'rot_diffusion': np.ones(compile_n_particles) * ROTATIONAL_DIFFUSION,
        'max_curvity': MAX_CURVITY,
        'min_curvity': MIN_CURVITY,
        'mid_curvity': MID_CURVITY
    }

    run_payload_simulation(compile_params, light=LIGHT_NPZ)
    print("JIT compilation complete.\n")

    #####################################################
    # BUILD SIMULATION PARAMETERS                       #
    #####################################################

    params = {
        'n_particles': N_PARTICLES,
        'box_size': BOX_SIZE,
        'dt': DT,
        'n_steps': N_STEPS,
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
        'walls': WALLS,
        'v0': np.ones(N_PARTICLES) * PARTICLE_V0,
        'curvity': np.zeros(N_PARTICLES),
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

    result = run_payload_simulation(params, light=LIGHT_NPZ)

    (saved_positions, saved_orientations, saved_velocities,
     saved_payload_positions, saved_payload_velocities, saved_curvity,
     saved_polarity, saved_particle_scores, particle_scores, polarity,
     simulation_time, final_step) = result

    print(f"\nSimulation completed in {simulation_time:.2f} seconds")
    print(f"Final step: {final_step}")

    # Machine-readable summary line for the results aggregator
    goal_reached = final_step < N_STEPS
    print(f"RESULT_LINE: G={MAZE_GRID_SIZE} M={M} prune_step={prune_step} "
          f"walls_remaining={walls_remaining} prune_seed={args.prune_seed} "
          f"final_step={final_step} goal_reached={goal_reached} "
          f"sim_time={simulation_time:.2f} output={args.output_name}")

    # Save data if enabled
    if SAVE_DATA:
        from src.runner import save_simulation_data
        if LIGHT_NPZ:
            save_light_simulation_data(
                DATA_OUTPUT_PATH,
                saved_positions, saved_payload_positions,
                saved_curvity, saved_particle_scores, params
            )
        else:
            save_simulation_data(
                DATA_OUTPUT_PATH,
                saved_positions, saved_orientations, saved_velocities,
                saved_payload_positions, saved_payload_velocities,
                params, saved_curvity, saved_polarity, saved_particle_scores
            )
        print(f"Data saved to: {DATA_OUTPUT_PATH}")

    if CREATE_VIDEO:
        create_payload_animation(
            saved_positions, saved_orientations, saved_velocities,
            saved_payload_positions, params, saved_curvity,
            output_file=OUTPUT_FILENAME,
            show_vectors=SHOW_VECTORS,
            polarity=saved_polarity,
            particle_scores=saved_particle_scores if COLOR_BY_SCORE else None
        )

    print("Simulation and visualization completed successfully!")
