"""Render the final frame of a simulation NPZ file as a static image.

Shows particle positions at the last saved step, the full payload trajectory,
walls, and the goal marker. Much faster than rendering a full video because
only the last frame of particle data is read from disk.
"""

import colorsys
import os
import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


SOURCE_FILE  = "D:/snell_9000_30_2m.npz"
OUTPUT_IMAGE = "D:/PostThesis/visualizations/snell_9000_30_2m_final.png"
COLOR_BY_SCORE = False   # True = rainbow by score, False = curvity blue/red


def _arc_points(x1, y1, x2, y2, c, n_pts=64):
    if abs(c) < 1e-9:
        return [x1, x2], [y1, y2]
    chx, chy = x2 - x1, y2 - y1
    chord_len = np.sqrt(chx**2 + chy**2)
    if chord_len < 1e-10:
        return [x1, x2], [y1, y2]
    R = chord_len / (2.0 * abs(c))
    h = np.sqrt(max(R**2 - (chord_len / 2)**2, 0.0))
    cw_perp = np.array([chy, -chx]) / chord_len
    mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    sc = 1.0 if c > 0 else -1.0
    center = mid + sc * h * cw_perp
    theta1 = np.arctan2(y1 - center[1], x1 - center[0])
    theta2 = np.arctan2(y2 - center[1], x2 - center[0])
    span_ccw = (theta2 - theta1) % (2 * np.pi)
    if span_ccw <= np.pi:
        thetas = np.linspace(theta1, theta1 + span_ccw, n_pts)
    else:
        span_cw = 2 * np.pi - span_ccw
        thetas = np.linspace(theta1, theta1 - span_cw, n_pts)
    return (center[0] + R * np.cos(thetas)).tolist(), (center[1] + R * np.sin(thetas)).tolist()


def curvity_color(c):
    c = float(np.clip(c, -1, 1))
    if c < 0:
        t = c + 1
        return (t * 0.5, t * 0.5, 0.5)
    else:
        return (0.5 + c * 0.5, 0.5 - c * 0.5, 0.5 - c * 0.5)


def score_color(s):
    hue = (0.75 + (float(s) % 50) / 50.0) % 1.0
    return colorsys.hsv_to_rgb(hue, 1.0, 1.0)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)

    print(f"Loading {SOURCE_FILE} ...")
    t0 = time.time()
    data = np.load(SOURCE_FILE, mmap_mode='r')

    # Read only what we need
    last_positions      = np.array(data['positions'][-1])       # (N, 2)
    payload_trajectory  = np.array(data['payload_positions'])   # (T, 2) — small
    last_curvity        = np.array(data['curvity_values'][-1])  # (N,)
    last_scores         = np.array(data['particle_scores'][-1]) # (N,)

    box_size      = float(data['box_size'])
    payload_radius = float(data['payload_radius'])
    particle_radius = np.array(data['particle_radius'])
    goal_position  = np.array(data['goal_position'])
    walls          = np.array(data['walls'])
    n_particles    = last_positions.shape[0]

    print(f"  {n_particles} particles, {payload_trajectory.shape[0]} trajectory frames  ({time.time()-t0:.1f}s)")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_aspect('equal')
    ax.set_title(f'Final frame — {n_particles} particles', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Walls
    for i in range(walls.shape[0]):
        xs, ys = _arc_points(walls[i,0], walls[i,1], walls[i,2], walls[i,3], walls[i,4])
        ax.plot(xs, ys, color='black', linewidth=2, solid_capstyle='round', zorder=10)

    # Payload trajectory
    ax.plot(payload_trajectory[:, 0], payload_trajectory[:, 1],
            color='#31DC13', linewidth=1.5, alpha=0.85, zorder=5, label='Payload path')

    # Particles (final frame)
    if COLOR_BY_SCORE:
        colors = [score_color(s) for s in last_scores]
    else:
        colors = [curvity_color(c) for c in last_curvity]

    r = float(particle_radius[0]) if particle_radius.ndim > 0 else float(particle_radius)
    ax.scatter(last_positions[:, 0], last_positions[:, 1],
               s=np.pi * (r * 1)**2, c=colors, alpha=0.7, zorder=6)

    # Payload (final position)
    payload_patch = mpatches.Circle(
        (payload_trajectory[-1, 0], payload_trajectory[-1, 1]),
        radius=payload_radius, color='gray', alpha=0.8, zorder=7
    )
    ax.add_patch(payload_patch)

    # Goal
    ax.plot(goal_position[0], goal_position[1],
            'g*', markersize=18, markeredgewidth=1.5, markeredgecolor='darkgreen', zorder=11)

    ax.legend(loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved to {OUTPUT_IMAGE}  (total: {time.time()-t0:.1f}s)")
