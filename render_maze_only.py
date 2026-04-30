"""Render a short MP4 of just the 30x30 maze + payload (no particles, no simulation)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from src import wilson

# ── Parameters ──────────────────────────────────────────────────────────────

BOX_SIZE        = 600
MAZE_GRID_SIZE  = 30
PAYLOAD_RADIUS  = 5
RANDOM_SEED     = 42

CELL_SIZE       = BOX_SIZE / MAZE_GRID_SIZE
PAYLOAD_START   = np.array([CELL_SIZE / 2, CELL_SIZE / 2])
GOAL_POSITION   = np.array([BOX_SIZE - CELL_SIZE / 2, BOX_SIZE - CELL_SIZE / 2])

OUTPUT_FILE     = "maze_preview.mp4"   # root folder
N_FRAMES        = 30                   # ~2 s at 15 fps — "extremely short"
FPS             = 15

# ── Build walls ──────────────────────────────────────────────────────────────

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


# ── Build maze ───────────────────────────────────────────────────────────────

np.random.seed(RANDOM_SEED)
passages = wilson.generate(MAZE_GRID_SIZE, seed=RANDOM_SEED)
walls    = maze_to_walls(passages, MAZE_GRID_SIZE, BOX_SIZE)

# ── Draw ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, BOX_SIZE)
ax.set_ylim(0, BOX_SIZE)
ax.set_aspect('equal')
ax.set_title(f'30×30 Maze — payload only, no particles', fontsize=13)
ax.grid(True, alpha=0.2)

for i in range(walls.shape[0]):
    xs, ys = arc_points(walls[i, 0], walls[i, 1], walls[i, 2], walls[i, 3], walls[i, 4])
    ax.plot(xs, ys, color='black', linewidth=1.5, solid_capstyle='round', zorder=10)

payload_patch = Circle(PAYLOAD_START, PAYLOAD_RADIUS, color='gray', alpha=0.8, zorder=5)
ax.add_patch(payload_patch)

ax.plot(GOAL_POSITION[0], GOAL_POSITION[1], 'g*',
        markersize=15, markeredgewidth=1.5, markeredgecolor='darkgreen', zorder=6)

time_text = ax.text(0.02, 0.98, 'Frame: 0', transform=ax.transAxes,
                    fontsize=11, verticalalignment='top')

all_artists = [payload_patch, time_text]

def init():
    return all_artists

def update(frame):
    time_text.set_text(f'Frame: {frame}')
    return all_artists

anim = FuncAnimation(fig, update, frames=N_FRAMES, init_func=init, blit=True, interval=66)

writer = FFMpegWriter(fps=FPS, bitrate=4000, codec='libx264',
                      extra_args=['-pix_fmt', 'yuv420p', '-crf', '18'])

print(f"Saving {OUTPUT_FILE} …")
anim.save(OUTPUT_FILE, writer=writer)
plt.close()
print(f"Done — saved to: {os.path.abspath(OUTPUT_FILE)}")
