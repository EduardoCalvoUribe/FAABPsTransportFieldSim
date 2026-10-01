"""Render overlaid payload paths for default vs optimal runs, with DBA averages.

Shows the simulation environment from a reference NPZ, then overlays all 5
default and 5 optimal payload trajectories together with their per-group DTW
Barycenter Average (DBA) paths.  A legend is placed below the inset.

All other rendering logic (walls, particles, inset, goal, etc.) is identical
to render_final_frame.py.
"""

import os
import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.patheffects as mpe
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
from tslearn.barycenters import dtw_barycenter_averaging


# ── Hyperparameters ──────────────────────────────────────────────────────────
# Reference file: supplies particles, walls, box geometry, goal
SOURCE_FILE      = "optimal_G20_run1.npz"
OUTPUT_IMAGE     = "visualizations/optimality_comparison.png"
COLOR_BY_SCORE   = True     # True = gradient by score, False = curvity blue/red
DRAW_OPENED_WALL = True    # True = draw red line where shortcut wall was removed

# Split output: render two PNGs instead of one
#   OUTPUT_IMAGE_DEFAULT → blue (default) paths only, no red opened-wall line
#   OUTPUT_IMAGE_OPTIMAL → red (optimal) paths only, with opened-wall line
SPLIT_OUTPUT         = False
OUTPUT_IMAGE_DEFAULT = "visualizations/optimality_default.png"
OUTPUT_IMAGE_OPTIMAL = "visualizations/optimality_optimal.png"

# DBA parameters
DBA_BARYCENTER_SIZE = 100
DBA_MAX_ITER        = 10

# Input file groups
DEFAULT_FILES = [
    "default_opt_G20_run1.npz",
    "default_opt_G20_run2.npz",
    "default_opt_G20_run3.npz",
    "default_opt_G20_run4.npz",
    "default_opt_G20_run5.npz",
]
OPTIMAL_FILES = [
    "optimal_G20_run1.npz",
    "optimal_G20_run2.npz",
    "optimal_G20_run3.npz",
    "optimal_G20_run4.npz",
    "optimal_G20_run5.npz",
]

# Colors
DARK_BLUE     = '#00008B'
RED_COLOR     = '#D62728'
PAYLOAD_COLOR = '#737BA8'
GOAL_COLOR    = '#008F25'   # dark green

# Score gradient: amber (score 0 = direct LoS to goal) → deep purple (far from goal)
# Both ends are clearly distinct from the blue/red paths and green goal.
SCORE_CMAP_LOW  = '#F4D03F'   # warm amber  — best (score 0)
SCORE_CMAP_HIGH = '#8E44AD'   # deep purple — worst (highest score)


# ── Helper functions (unchanged from render_final_frame.py) ──────────────────

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
    """Neg curvity → #F46036, zero → light gray, pos curvity → #007FFF (azure)."""
    c = float(np.clip(c, -1, 1))
    gray = (0.62, 0.62, 0.62)
    if c < 0:
        t = c + 1   # 0 at c=-1, 1 at c=0
        # #FF5500 = (1.0, 0.333, 0.0) — neon orange
        return (t * gray[0] + (1 - t) * 1.000,
                t * gray[1] + (1 - t) * 0.333,
                t * gray[2] + (1 - t) * 0.000)
    else:
        t = c       # 0 at c=0, 1 at c=+1
        # #00AAFF = (0.0, 0.667, 1.0) — neon azure
        return (gray[0] + t * (0.000 - gray[0]),
                gray[1] + t * (0.667 - gray[1]),
                gray[2] + t * (1.000 - gray[2]))



# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)

    # ── Load reference environment ────────────────────────────────────────────
    print(f"Loading reference environment from {SOURCE_FILE} ...")
    t0 = time.time()
    data = np.load(SOURCE_FILE, mmap_mode='r')

    last_positions   = np.array(data['positions'][-1])       # (N, 2)
    last_curvity     = np.array(data['curvity_values'][-1])  # (N,)
    last_scores      = np.array(data['particle_scores'][-1]) # (N,)
    score_max_global = max(int(data['particle_scores'].max()), 1)  # true max across all timesteps
    ref_payload_traj = np.array(data['payload_positions'])   # (T, 2) — for inset

    box_size        = float(data['box_size'])
    payload_radius  = float(data['payload_radius'])
    particle_radius = np.array(data['particle_radius'])
    goal_position   = np.array(data['goal_position'])
    walls           = np.array(data['walls'])
    n_particles     = last_positions.shape[0]

    # Opened wall segment
    opened_wall_seg = None
    if 'opened_wall_segment' in data:
        ows = np.array(data['opened_wall_segment'])
        if np.any(ows != 0):
            opened_wall_seg = ows
    else:
        try:
            import optimal_path as _op
            from src import wilson as _wil
            _gs = int(round(box_size / 60.0))
            if abs(_gs * 60.0 - box_size) < 0.5 and _gs >= 2:
                _cs   = box_size / _gs
                _base = _wil.generate(_gs, seed=42)
                _cands, _ = _op.find_shortcut_walls(
                    _base, _gs, (0, 0), (_gs - 1, _gs - 1))
                if _cands:
                    _, _, _cA, _cB = _cands[0]
                    (rA, cA_), (rB, cB_) = _cA, _cB
                    if rA == rB:
                        wx = (min(cA_, cB_) + 1) * _cs
                        opened_wall_seg = np.array([wx, rA*_cs, wx, (rA+1)*_cs])
                    else:
                        wy = (min(rA, rB) + 1) * _cs
                        opened_wall_seg = np.array([cA_*_cs, wy, (cA_+1)*_cs, wy])
        except Exception:
            pass

    print(f"  {n_particles} particles  ({time.time()-t0:.1f}s)")

    # ── Load payload trajectories ─────────────────────────────────────────────
    print("Loading payload trajectories ...")
    default_paths = []
    for f in DEFAULT_FILES:
        d = np.load(f, mmap_mode='r')
        default_paths.append(np.array(d['payload_positions'], dtype=np.float64))
        print(f"  {f}: {default_paths[-1].shape[0]} steps")

    optimal_paths = []
    for f in OPTIMAL_FILES:
        d = np.load(f, mmap_mode='r')
        optimal_paths.append(np.array(d['payload_positions'], dtype=np.float64))
        print(f"  {f}: {optimal_paths[-1].shape[0]} steps")

    # ── Compute DBA averages ──────────────────────────────────────────────────
    print(f"Computing DBA average (default group, barycenter_size={DBA_BARYCENTER_SIZE}) ...")
    t1 = time.time()
    default_avg = dtw_barycenter_averaging(
        default_paths,
        barycenter_size=DBA_BARYCENTER_SIZE,
        max_iter=DBA_MAX_ITER,
        verbose=True,
    )  # shape (DBA_BARYCENTER_SIZE, 2)
    print(f"  done in {time.time()-t1:.1f}s")

    print(f"Computing DBA average (optimal group, barycenter_size={DBA_BARYCENTER_SIZE}) ...")
    t1 = time.time()
    optimal_avg = dtw_barycenter_averaging(
        optimal_paths,
        barycenter_size=DBA_BARYCENTER_SIZE,
        max_iter=DBA_MAX_ITER,
        verbose=True,
    )  # shape (DBA_BARYCENTER_SIZE, 2)
    print(f"  done in {time.time()-t1:.1f}s")

    # Final positions of each average path
    default_avg_end = default_avg[-1]   # (2,)
    optimal_avg_end = optimal_avg[-1]   # (2,)

    # px_final / py_final from reference file (for inset centering)
    px_final = ref_payload_traj[-1, 0]
    py_final = ref_payload_traj[-1, 1]

    # ── Render helper ─────────────────────────────────────────────────────────
    def render_figure(*, draw_default, draw_optimal, draw_opened_wall, output_path):
        """Create and save one PNG.

        draw_default      – include blue (default) paths and payload marker
        draw_optimal      – include red (optimal) paths and payload marker
        draw_opened_wall  – draw the red shortcut-wall line
        output_path       – destination file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.set_aspect('equal')
        ax.axis('off')

        # Walls
        wall_lw = 1.3 if box_size > 1200 else 1.8
        for i in range(walls.shape[0]):
            xs, ys = _arc_points(walls[i,0], walls[i,1], walls[i,2], walls[i,3], walls[i,4])
            ax.plot(xs, ys, color='#505050', linewidth=wall_lw, solid_capstyle='round', zorder=10)

        # Opened wall — red solid line where the shortcut wall was removed
        if draw_opened_wall and opened_wall_seg is not None:
            ax.plot([opened_wall_seg[0], opened_wall_seg[2]],
                    [opened_wall_seg[1], opened_wall_seg[3]],
                    color=GOAL_COLOR, linewidth=3.0, linestyle='-',
                    solid_capstyle='round', zorder=11,
                    path_effects=[mpe.Stroke(linewidth=4.5, foreground='black'), mpe.Normal()])

        # ── Payload paths (back to front) ─────────────────────────────────────
        # 1. Default individual paths — dark blue, low alpha, thin
        if draw_default:
            for path in default_paths:
                ax.plot(path[:, 0], path[:, 1],
                        color=DARK_BLUE, alpha=0.30, linewidth=1.2,
                        linestyle='-', zorder=2)

        # 2. Optimal individual paths — red, low alpha, thin
        if draw_optimal:
            for path in optimal_paths:
                ax.plot(path[:, 0], path[:, 1],
                        color=RED_COLOR, alpha=0.30, linewidth=1.2,
                        linestyle='-', zorder=3)

        # 3. Default average path — dark blue, full alpha, thicker
        if draw_default:
            ax.plot(default_avg[:, 0], default_avg[:, 1],
                    color=DARK_BLUE, alpha=1.0, linewidth=2.5,
                    linestyle='-', zorder=4)

        # 4. Optimal average path — red, full alpha, thicker
        if draw_optimal:
            ax.plot(optimal_avg[:, 0], optimal_avg[:, 1],
                    color=RED_COLOR, alpha=1.0, linewidth=2.5,
                    linestyle='-', zorder=5)

        # ── Particles (final frame of reference file) ──────────────────────────
        if COLOR_BY_SCORE:
            score_cmap = LinearSegmentedColormap.from_list(
                'score', [SCORE_CMAP_LOW, SCORE_CMAP_HIGH])
            score_norm = Normalize(vmin=0, vmax=score_max_global)
            colors = [score_cmap(score_norm(s))[:3] for s in last_scores]
        else:
            colors = [curvity_color(c) for c in last_curvity]

        r = float(particle_radius[0]) if particle_radius.ndim > 0 else float(particle_radius)
        if box_size > 1200:
            min_r = box_size / 600.0
        elif box_size > 600:
            min_r = box_size / 400.0
        else:
            min_r = box_size / 300.0
        display_radii = np.maximum(particle_radius, min_r)
        circles = [mpatches.Circle((x, y), radius=ri)
                   for (x, y), ri in zip(last_positions, display_radii)]
        pc = PatchCollection(circles, facecolors=colors, alpha=0.7, linewidths=0, zorder=6)
        ax.add_collection(pc)

        # For small maps (G2) label every particle in the main view
        if box_size <= 120:
            label_offset_main = float(display_radii[0]) * 1.4
            for i in range(n_particles):
                ax.text(last_positions[i, 0] - label_offset_main, last_positions[i, 1],
                        str(i), fontsize=8, ha='right', va='center',
                        color='black', zorder=9)

        # 5. Payload markers — one per group, at the final point of each average path
        if draw_default:
            ax.add_patch(mpatches.Circle(
                (default_avg_end[0], default_avg_end[1]), radius=payload_radius,
                facecolor=DARK_BLUE, edgecolor='none',
                alpha=0.5, zorder=7,
            ))
        if draw_optimal:
            ax.add_patch(mpatches.Circle(
                (optimal_avg_end[0], optimal_avg_end[1]), radius=payload_radius,
                facecolor=RED_COLOR, edgecolor='none',
                alpha=0.5, zorder=7,
            ))

        # Goal
        ax.plot(goal_position[0], goal_position[1],
                '*', color=GOAL_COLOR, markersize=18, markeredgewidth=0.8,
                markeredgecolor='#003A0E', zorder=11)

        # ── Fill figure exactly, then place inset ─────────────────────────────
        ax.set_position([0, 0, 1, 1])

        # ── Inset: zoomed payload view — only for G10+ (box_size > 300) ───────
        if box_size > 300:
            zoom_half = 40
            _ap = ax.get_position()
            inset_ax = fig.add_axes([
                _ap.x0 + 0.02 * _ap.width,
                _ap.y0 + 0.73 * _ap.height,
                0.25 * _ap.width,
                0.25 * _ap.height,
            ])
            inset_ax.set_xlim(-zoom_half, zoom_half)
            inset_ax.set_ylim(-zoom_half, zoom_half)
            inset_ax.set_aspect('equal')
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_title('Payload zoom', fontsize=7, pad=2)
            for spine in inset_ax.spines.values():
                spine.set_linewidth(3)
            inset_ax.set_facecolor('white')

            # Particle positions relative to final payload centre (reference file)
            rel_x = last_positions[:, 0] - px_final
            rel_y = last_positions[:, 1] - py_final
            inset_circles = [mpatches.Circle((dx, dy), radius=ri)
                             for dx, dy, ri in zip(rel_x, rel_y, particle_radius)]
            inset_pc = PatchCollection(inset_circles, facecolors=colors, alpha=0.7, linewidths=0, zorder=6)
            inset_ax.add_collection(inset_pc)

            # Particle ID labels — shown only for particles inside the visible zoom window
            label_offset = r * 1.4
            for i in range(n_particles):
                if abs(rel_x[i]) < zoom_half and abs(rel_y[i]) < zoom_half:
                    inset_ax.text(rel_x[i] - label_offset, rel_y[i],
                                  str(i), fontsize=6, ha='right', va='center',
                                  color='black', zorder=9)

            # Payload markers for each group average — relative to reference px_final/py_final
            if draw_default:
                inset_ax.add_patch(mpatches.Circle(
                    (default_avg_end[0] - px_final, default_avg_end[1] - py_final),
                    radius=payload_radius,
                    facecolor=DARK_BLUE, edgecolor='none',
                    alpha=0.5, zorder=7,
                ))
            if draw_optimal:
                inset_ax.add_patch(mpatches.Circle(
                    (optimal_avg_end[0] - px_final, optimal_avg_end[1] - py_final),
                    radius=payload_radius,
                    facecolor=RED_COLOR, edgecolor='none',
                    alpha=0.5, zorder=7,
                ))

            # Goal in payload-relative coordinates
            gx_rel = goal_position[0] - px_final
            gy_rel = goal_position[1] - py_final
            inset_ax.plot(gx_rel, gy_rel, '*', color=GOAL_COLOR, markersize=24,
                          markeredgewidth=0.8, markeredgecolor='#003A0E', zorder=100)

            # Walls in payload-relative coordinates
            for i in range(walls.shape[0]):
                wx1 = walls[i, 0] - px_final
                wy1 = walls[i, 1] - py_final
                wx2 = walls[i, 2] - px_final
                wy2 = walls[i, 3] - py_final
                xs, ys = _arc_points(wx1, wy1, wx2, wy2, walls[i, 4])
                inset_ax.plot(xs, ys, color='#505050', linewidth=1.5, solid_capstyle='round', zorder=10)

            # Opened wall in payload-relative coordinates
            if draw_opened_wall and opened_wall_seg is not None:
                inset_ax.plot(
                    [opened_wall_seg[0] - px_final, opened_wall_seg[2] - px_final],
                    [opened_wall_seg[1] - py_final, opened_wall_seg[3] - py_final],
                    color=GOAL_COLOR, linewidth=2.5, linestyle='-',
                    solid_capstyle='round', zorder=11,
                    path_effects=[mpe.Stroke(linewidth=4.0, foreground='black'), mpe.Normal()])

            # Average paths relative to reference px_final/py_final
            if draw_default:
                inset_ax.plot(default_avg[:, 0] - px_final, default_avg[:, 1] - py_final,
                              color=DARK_BLUE, alpha=1.0, linewidth=2.5, linestyle='-', zorder=4)
            if draw_optimal:
                inset_ax.plot(optimal_avg[:, 0] - px_final, optimal_avg[:, 1] - py_final,
                              color=RED_COLOR, alpha=1.0, linewidth=2.5, linestyle='-', zorder=5)

        # ── Legend — placed below the inset (top-left quadrant) ───────────────
        legend_handles = []
        if draw_default:
            legend_handles += [
                mlines.Line2D([], [], color=DARK_BLUE, alpha=0.30, linewidth=1.5,
                              label='default runs'),
                mlines.Line2D([], [], color=DARK_BLUE, alpha=1.0,  linewidth=2.5,
                              label='default average'),
            ]
        if draw_optimal:
            legend_handles += [
                mlines.Line2D([], [], color=RED_COLOR, alpha=0.30, linewidth=1.5,
                              label='optimal runs'),
                mlines.Line2D([], [], color=RED_COLOR, alpha=1.0,  linewidth=2.5,
                              label='optimal average'),
            ]
        leg = ax.legend(
            handles=legend_handles,
            loc='upper left',
            bbox_to_anchor=(0.02, 0.71),
            fontsize=9,
            frameon=True,
            edgecolor='#aaaaaa',
        )
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_alpha(1.0)
        leg.set_zorder(20)

        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"  Saved {output_path}")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if SPLIT_OUTPUT:
        render_figure(draw_default=True,  draw_optimal=False,
                      draw_opened_wall=False, output_path=OUTPUT_IMAGE_DEFAULT)
        render_figure(draw_default=False, draw_optimal=True,
                      draw_opened_wall=DRAW_OPENED_WALL, output_path=OUTPUT_IMAGE_OPTIMAL)
    else:
        render_figure(draw_default=True, draw_optimal=True,
                      draw_opened_wall=DRAW_OPENED_WALL, output_path=OUTPUT_IMAGE)

    print(f"Done  (total: {time.time()-t0:.1f}s)")
