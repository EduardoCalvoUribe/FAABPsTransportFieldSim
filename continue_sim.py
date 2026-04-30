"""Continue a payload-transport simulation from the last frame of a light NPZ.

Loads SOURCE_NPZ, picks up particle positions, scores, and the existing payload
trajectory, then runs N_STEPS_CONT more steps.  The final frame is rendered with
the original payload path (green) and the continuation path (orange) overlaid.

Usage
-----
Edit SOURCE_NPZ, OUTPUT_NPZ, OUTPUT_IMAGE, N_STEPS_CONT, and (if needed) the
simulation parameters below, then run:

    python continue_sim.py

The simulation params section must match the original run.  The script re-uses
the maze and all physics constants from main.py via `import main`.

Orientation and polarity are NOT stored in a light NPZ, so they are
re-initialised at random / default values.  Polarity is recalculated on the
first score_and_polarity_update_interval step (default: step 20), and
orientations equilibrate quickly through the nudge mechanism.
"""

import colorsys
import os
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import main as _main  # reuse maze + all physics constants; does NOT run the sim
from src.runner import (
    run_payload_simulation,
    run_payload_simulation_from_state,
    save_light_simulation_data,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
SOURCE_NPZ   = "data/snell_9000_30_2m.npz"         # light NPZ from the original run
OUTPUT_NPZ   = "data/snell_9000_30_cont.npz"        # combined-path NPZ to write
OUTPUT_IMAGE = "visualizations/snell_9000_30_cont_final.png"

# ─── How many additional steps ────────────────────────────────────────────────
N_STEPS_CONT = 500_000

# ─── Save combined NPZ ────────────────────────────────────────────────────────
SAVE_DATA = True

# ─── Rendering ────────────────────────────────────────────────────────────────
COLOR_BY_SCORE = False   # True = rainbow by score, False = curvity blue/red


# ─── Load last-frame state from a light NPZ ──────────────────────────────────

def load_last_frame(npz_path):
    """Return last-frame state and full payload trajectory from a light NPZ.

    Returns
    -------
    positions        : (N, 2)  particle positions at the last saved frame
    payload_traj     : (T, 2)  complete payload trajectory from the original run
    particle_scores  : (N,)    scores at the last saved frame
    box_size, payload_radius, particle_radius, goal_position, walls
    """
    print(f"Loading {npz_path} ...")
    t0 = time.time()
    data = np.load(npz_path)
    positions       = np.array(data['positions'][-1])         # (N, 2)
    payload_traj    = np.array(data['payload_positions'])      # (T, 2)
    particle_scores = np.array(data['particle_scores'][-1])   # (N,)
    box_size        = float(data['box_size'])
    payload_radius  = float(data['payload_radius'])
    particle_radius = np.array(data['particle_radius'])
    goal_position   = np.array(data['goal_position'])
    walls           = np.array(data['walls'])
    print(f"  {positions.shape[0]} particles, {payload_traj.shape[0]} trajectory frames  "
          f"({time.time() - t0:.1f}s)")
    return positions, payload_traj, particle_scores, box_size, payload_radius, particle_radius, goal_position, walls


# ─── Rendering helpers ────────────────────────────────────────────────────────

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


def _curvity_color(c):
    c = float(np.clip(c, -1, 1))
    if c < 0:
        t = c + 1
        return (t * 0.5, t * 0.5, 0.5)
    return (0.5 + c * 0.5, 0.5 - c * 0.5, 0.5 - c * 0.5)


def _score_color(s):
    hue = (0.75 + (float(s) % 50) / 50.0) % 1.0
    return colorsys.hsv_to_rgb(hue, 1.0, 1.0)


def render_combined_final_frame(
    output_image,
    last_positions, last_curvity, last_scores,
    path_a, path_b,
    box_size, payload_radius, particle_radius,
    goal_position, walls,
    color_by_score=False,
):
    """Render the final frame with the original path (A) and continuation path (B).

    path_a : (T1, 2)  payload trajectory from the original run
    path_b : (T2, 2)  payload trajectory from the continuation run
    Both are drawn starting from their own first point; the junction is where
    path_a ends and path_b begins (same coordinate, so they connect seamlessly).
    """
    output_dir = os.path.dirname(output_image)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_aspect('equal')
    ax.set_title(f'Final frame (A+B) — {last_positions.shape[0]} particles', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Walls
    for i in range(walls.shape[0]):
        xs, ys = _arc_points(walls[i, 0], walls[i, 1], walls[i, 2], walls[i, 3], walls[i, 4])
        ax.plot(xs, ys, color='black', linewidth=2, solid_capstyle='round', zorder=10)

    # Combined payload path (A then B) as a single green line
    combined = np.concatenate([path_a, path_b[1:]], axis=0)
    ax.plot(combined[:, 0], combined[:, 1],
            color='#31DC13', linewidth=1.5, alpha=0.85, zorder=5)

    # Particles at final frame
    if color_by_score:
        colors = [_score_color(s) for s in last_scores]
    else:
        colors = [_curvity_color(c) for c in last_curvity]

    r = float(particle_radius[0]) if particle_radius.ndim > 0 else float(particle_radius)
    ax.scatter(last_positions[:, 0], last_positions[:, 1],
               s=np.pi * (r * 1)**2, c=colors, alpha=0.7, zorder=6)

    # Payload circle at final position
    ax.add_patch(mpatches.Circle(
        (path_b[-1, 0], path_b[-1, 1]),
        radius=payload_radius, color='gray', alpha=0.8, zorder=7,
    ))

    # Goal marker
    ax.plot(goal_position[0], goal_position[1],
            'g*', markersize=18, markeredgewidth=1.5, markeredgecolor='darkgreen', zorder=11)

    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_image}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(_main.RANDOM_SEED)

    # ── Load last frame ───────────────────────────────────────────────────────
    (init_positions, path_a, init_scores,
     box_size, payload_radius, particle_radius,
     goal_position, walls) = load_last_frame(SOURCE_NPZ)

    n_particles = init_positions.shape[0]

    # Orientations not stored in light NPZ — initialise randomly.
    init_orientations = np.zeros((n_particles, 2))
    angles = np.random.uniform(0, 2 * np.pi, n_particles)
    init_orientations[:, 0] = np.cos(angles)
    init_orientations[:, 1] = np.sin(angles)

    # Polarity defaults to π/4; recalculated on the first polarity update step.
    init_polarity = np.empty((n_particles, 2))
    init_polarity[:, 0] = np.cos(np.pi / 4)
    init_polarity[:, 1] = np.sin(np.pi / 4)

    # Resume from the last saved payload position.
    init_payload_pos = path_a[-1].copy()

    # ── Build params (must match the original run) ────────────────────────────
    params = {
        'n_particles':  n_particles,
        'box_size':     _main.BOX_SIZE,
        'dt':           _main.DT,
        'n_steps':      N_STEPS_CONT,
        'save_interval': _main.SAVE_INTERVAL,

        'payload_radius':   _main.PAYLOAD_RADIUS,
        'payload_mobility': _main.PAYLOAD_MOBILITY,
        'stiffness':        _main.STIFFNESS,

        'goal_position':                    _main.GOAL_POSITION,
        'particle_view_range':              _main.PARTICLE_VIEW_RANGE,
        'score_and_polarity_update_interval': _main.SCORE_AND_POLARITY_UPDATE_INTERVAL,
        'end_when_goal_reached':            _main.END_WHEN_GOAL_REACHED,
        'polarity_nudge_interval':          _main.POLARITY_NUDGE_INTERVAL,
        'polarity_nudge_strength':          _main.POLARITY_NUDGE_STRENGTH,

        'walls': _main.WALLS if _main.WALLS is not None else np.zeros((0, 5), dtype=np.float64),

        'v0':            np.ones(n_particles) * _main.PARTICLE_V0,
        'curvity':       np.zeros(n_particles),
        'particle_radius': np.ones(n_particles) * _main.PARTICLE_RADIUS,
        'mobility':      np.ones(n_particles) * _main.PARTICLE_MOBILITY,
        'rot_diffusion': np.ones(n_particles) * _main.ROTATIONAL_DIFFUSION,

        'max_curvity': _main.MAX_CURVITY,
        'min_curvity': _main.MIN_CURVITY,
        'mid_curvity': _main.MID_CURVITY,
    }

    # ── JIT warm-up ───────────────────────────────────────────────────────────
    print("Warming up JIT (10-particle test run)...")
    compile_params = dict(params)
    compile_params['n_particles']      = 10
    compile_params['n_steps']          = 10
    compile_params['payload_position'] = init_payload_pos.copy()
    for key in ('v0', 'curvity', 'particle_radius', 'mobility', 'rot_diffusion'):
        compile_params[key] = np.ones(10) * params[key][0]
    run_payload_simulation(compile_params, light=True)
    print("JIT warm-up done.\n")

    # ── Run continuation ──────────────────────────────────────────────────────
    result = run_payload_simulation_from_state(
        params,
        initial_positions=init_positions,
        initial_orientations=init_orientations,
        initial_payload_pos=init_payload_pos,
        initial_particle_scores=init_scores,
        initial_polarity=init_polarity,
        light=True,
    )

    (saved_positions, _, _, path_b, _,
     saved_curvity, _, saved_particle_scores,
     _, _, sim_time, final_step) = result

    print(f"\nContinuation done in {sim_time:.2f}s  (step {final_step})")

    # ── Combine payload paths ─────────────────────────────────────────────────
    # path_b[0] == init_payload_pos == path_a[-1], so drop the duplicate.
    combined_path = np.concatenate([path_a, path_b[1:]], axis=0)

    # ── Save combined light NPZ ───────────────────────────────────────────────
    if SAVE_DATA:
        os.makedirs(os.path.dirname(OUTPUT_NPZ) or ".", exist_ok=True)
        save_light_simulation_data(
            OUTPUT_NPZ,
            saved_positions,
            combined_path,
            saved_curvity,
            saved_particle_scores,
            params,
        )
        print(f"Combined NPZ saved to: {OUTPUT_NPZ}")

    # ── Render ────────────────────────────────────────────────────────────────
    render_combined_final_frame(
        OUTPUT_IMAGE,
        last_positions=saved_positions[-1],
        last_curvity=saved_curvity[-1],
        last_scores=saved_particle_scores[-1],
        path_a=path_a,
        path_b=path_b,
        box_size=_main.BOX_SIZE,
        payload_radius=_main.PAYLOAD_RADIUS,
        particle_radius=params['particle_radius'],
        goal_position=_main.GOAL_POSITION,
        walls=params['walls'],
        color_by_score=COLOR_BY_SCORE,
    )
