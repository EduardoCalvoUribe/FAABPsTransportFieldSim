"""Render a static PNG of just the 30x30 maze + payload & goal"""
import random
import os
from collections import defaultdict, deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ── Parameters ──────────────────────────────────────────────────────────────

BOX_SIZE        = 1200
MAZE_GRID_SIZE  = 10
PAYLOAD_RADIUS  = 5
RANDOM_SEED     = 42
DRAW_CURVED_CORNERS = True

CELL_SIZE       = BOX_SIZE / MAZE_GRID_SIZE
PAYLOAD_START   = np.array([CELL_SIZE / 2, CELL_SIZE / 2])
GOAL_POSITION   = np.array([BOX_SIZE - CELL_SIZE / 2, BOX_SIZE - CELL_SIZE / 2])

OUTPUT_FILE     = "visualizations/maze_G10_seed42.png"
DRAW_SOLUTION   = False

# ── Maze ──────────────────────────────────────────────────────────────

# Generate maze using Wilson's algorithm. Data structure:
#   dict[
#       (row, col),   # a room's address
#       set of (row, col)  # its open neighbors
#   ]
def wilson_generate(width, height=None, seed=None):
    if height is None:
        height = width
    rng = random.Random(seed)
    passages = defaultdict(set)

    def neighbors(r, c):
        result = []
        if r > 0:          result.append((r - 1, c))
        if r < height - 1: result.append((r + 1, c))
        if c > 0:          result.append((r, c - 1))
        if c < width - 1:  result.append((r, c + 1))
        return result

    all_cells = [(r, c) for r in range(height) for c in range(width)]
    in_tree = {all_cells[0]}
    not_in_tree = list(all_cells[1:])
    rng.shuffle(not_in_tree)

    for start in not_in_tree:
        if start in in_tree:
            continue
        path = [start]
        visited_order = {start: 0}
        current = start
        while current not in in_tree:
            nxt = rng.choice(neighbors(*current))
            if nxt in visited_order:
                loop_start = visited_order[nxt]
                for cell in path[loop_start + 1:]:
                    del visited_order[cell]
                path = path[:loop_start + 1]
            else:
                visited_order[nxt] = len(path)
                path.append(nxt)
            current = nxt
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            passages[a].add(b)
            passages[b].add(a)
            in_tree.add(a)
        in_tree.add(path[-1])

    return dict(passages)

# generate appropriate walls given maze
def maze_to_walls(passages, grid_size, box_size, include_boundary=True):
    cell   = box_size / grid_size
    R      = cell / 2
    C_ARC  = 0.707107
    walls  = []
    if include_boundary:
        walls += [
            [0,              R - 5,           0,              box_size - R + 5, 0],
            [cell / 2 - 5,  0,               box_size - R + 5, 0,             0],
            [box_size - R + 5, box_size,      R - 5,          box_size,        0],
            [box_size,       box_size - R + 5, box_size,       R,              0],
        ]
        walls += [
            [0,              R,               R,              0,               -C_ARC],
            [box_size - R,   0,               box_size,       R,               -C_ARC],
            [R,              box_size,        0,              box_size - R,    -C_ARC],
            [box_size - R,   box_size,        box_size,       box_size - R,     C_ARC],
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
    for r_int in range(1, grid_size):
        for c_int in range(1, grid_size):
            cx = c_int * cell;  ry = r_int * cell
            h_right = (r_int, c_int)     not in passages.get((r_int - 1, c_int),     set())
            h_left  = (r_int, c_int - 1) not in passages.get((r_int - 1, c_int - 1), set())
            v_up    = (r_int, c_int)     not in passages.get((r_int,     c_int - 1), set())
            v_down  = (r_int - 1, c_int) not in passages.get((r_int - 1, c_int - 1), set())
            if h_right and v_up:   walls.append([cx + R, ry, cx, ry + R,  C_ARC])
            if h_right and v_down: walls.append([cx + R, ry, cx, ry - R, -C_ARC])
            if h_left  and v_up:   walls.append([cx - R, ry, cx, ry + R, -C_ARC])
            if h_left  and v_down: walls.append([cx - R, ry, cx, ry - R,  C_ARC])
    if include_boundary:
        for r_int in range(1, grid_size):
            if (r_int, 0) not in passages.get((r_int - 1, 0), set()):
                ry = r_int * cell
                walls.append([R, ry, 0, ry + R,  C_ARC])
                walls.append([R, ry, 0, ry - R, -C_ARC])
        for r_int in range(1, grid_size):
            if (r_int, grid_size - 1) not in passages.get((r_int - 1, grid_size - 1), set()):
                ry = r_int * cell
                walls.append([box_size - R, ry, box_size, ry + R, -C_ARC])
                walls.append([box_size - R, ry, box_size, ry - R,  C_ARC])
        for c_int in range(1, grid_size):
            if (0, c_int) not in passages.get((0, c_int - 1), set()):
                cx = c_int * cell
                walls.append([cx + R, 0, cx, R,  C_ARC])
                walls.append([cx - R, 0, cx, R, -C_ARC])
        for c_int in range(1, grid_size):
            if (grid_size - 1, c_int) not in passages.get((grid_size - 1, c_int - 1), set()):
                cx = c_int * cell
                walls.append([cx + R, box_size, cx, box_size - R, -C_ARC])
                walls.append([cx - R, box_size, cx, box_size - R,  C_ARC])
    return np.array(walls, dtype=np.float64)

def arc_points(x1, y1, x2, y2, c, n_pts=64):
    if abs(c) < 1e-9:
        return [x1, x2], [y1, y2]
    chx, chy = x2 - x1, y2 - y1
    chord_len = np.sqrt(chx**2 + chy**2)
    if chord_len < 1e-10:
        return [x1, x2], [y1, y2]
    R    = chord_len / (2.0 * abs(c))
    h    = np.sqrt(max(R**2 - (chord_len / 2)**2, 0.0))
    perp = np.array([chy, -chx]) / chord_len
    mid  = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    sc   = 1.0 if c > 0 else -1.0
    ctr  = mid + sc * h * perp
    th1  = np.arctan2(y1 - ctr[1], x1 - ctr[0])
    th2  = np.arctan2(y2 - ctr[1], x2 - ctr[0])
    span_ccw = (th2 - th1) % (2 * np.pi)
    if span_ccw <= np.pi:
        thetas = np.linspace(th1, th1 + span_ccw, n_pts)
    else:
        thetas = np.linspace(th1, th1 - (2 * np.pi - span_ccw), n_pts)
    return (ctr[0] + R * np.cos(thetas)).tolist(), (ctr[1] + R * np.sin(thetas)).tolist()


def solve_maze_bfs(passages, start, goal):
    """Return list of (row, col) cells from start to goal via BFS."""
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        cell, path = queue.popleft()
        if cell == goal:
            return path
        for neighbor in passages.get(cell, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None


# ── Build maze ───────────────────────────────────────────────────────────────

np.random.seed(RANDOM_SEED)
passages = wilson_generate(MAZE_GRID_SIZE, seed=RANDOM_SEED)
walls    = maze_to_walls(passages, MAZE_GRID_SIZE, BOX_SIZE)

# ── Draw ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, BOX_SIZE)
ax.set_ylim(0, BOX_SIZE)
ax.set_aspect('equal')
ax.axis('off')

for i in range(walls.shape[0]):
    if not DRAW_CURVED_CORNERS and abs(walls[i, 4]) > 1e-9:
        continue
    xs, ys = arc_points(walls[i, 0], walls[i, 1], walls[i, 2], walls[i, 3], walls[i, 4])
    ax.plot(xs, ys, color='black', linewidth=1.5, solid_capstyle='round', zorder=10)

payload_patch = Circle(PAYLOAD_START, PAYLOAD_RADIUS, color='gray', alpha=0.8, zorder=5)
ax.add_patch(payload_patch)

ax.plot(GOAL_POSITION[0], GOAL_POSITION[1], 'g*',
        markersize=15, markeredgewidth=1.5, markeredgecolor='darkgreen', zorder=6)

if DRAW_SOLUTION:
    start_cell = (0, 0)
    goal_cell  = (MAZE_GRID_SIZE - 1, MAZE_GRID_SIZE - 1)
    solution   = solve_maze_bfs(passages, start_cell, goal_cell)
    if solution is not None:
        xs = [(c + 0.5) * CELL_SIZE for r, c in solution]
        ys = [(r + 0.5) * CELL_SIZE for r, c in solution]
        ax.plot(xs, ys, color='red', linewidth=1.5, alpha=0.7, zorder=7)
        path_length = (len(solution) - 1) * CELL_SIZE
        print(f"Solution path: {len(solution)} cells, {len(solution) - 1} steps, "
              f"length = {path_length:.2f} units")

print(f"Saving {OUTPUT_FILE} …")
fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', pad_inches=0)
plt.close()
print(f"Done — saved to: {os.path.abspath(OUTPUT_FILE)}")
