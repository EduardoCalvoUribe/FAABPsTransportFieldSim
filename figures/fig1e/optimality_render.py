"""Render overlaid payload paths for default vs optimal runs, with DBA averages.

Shows the simulation environment from a reference NPZ, then overlays all 5
default and 5 optimal payload trajectories together with their per-group DTW
Barycenter Average (DBA) paths.  A legend is placed in the top-left corner (no inset).

All other rendering logic (walls, particles, inset, goal, etc.) is identical
to render_final_frame.py.
"""
from pathlib import Path

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
BASEPATH         = Path(__file__).parent
SOURCE_FILE      = BASEPATH / "fig1e_data/optimal_G20_run2.npz"
OUTPUT_IMAGE     = BASEPATH / "optimality_comparison.png"
COLOR_BY_SCORE   = True     # True = gradient by score, False = curvity blue/red
DRAW_OPENED_WALL = True    # True = draw red line where shortcut wall was removed

# Split output: render two PNGs instead of one
#   OUTPUT_IMAGE_DEFAULT → blue (default) paths only, no red opened-wall line
#   OUTPUT_IMAGE_OPTIMAL → red (optimal) paths only, with opened-wall line
SPLIT_OUTPUT         = False
OUTPUT_IMAGE_DEFAULT = BASEPATH / "optimality_default2.png"
OUTPUT_IMAGE_OPTIMAL = BASEPATH / "optimality_optimal2.png"

# DBA parameters
DBA_BARYCENTER_SIZE = 100
DBA_MAX_ITER        = 10

# Input file groups
DEFAULT_FILES = [
    BASEPATH / "fig1e_data/default_opt_G20_run1.npz",
    BASEPATH / "fig1e_data/default_opt_G20_run2.npz",
    BASEPATH / "fig1e_data/default_opt_G20_run3.npz",
    BASEPATH / "fig1e_data/default_opt_G20_run4.npz",
    BASEPATH / "fig1e_data/default_opt_G20_run5.npz",
]
OPTIMAL_FILES = [
    BASEPATH / "fig1e_data/optimal_G20_run1.npz",
    BASEPATH / "fig1e_data/optimal_G20_run2.npz",
    BASEPATH / "fig1e_data/optimal_G20_run3.npz",
    BASEPATH / "fig1e_data/optimal_G20_run4.npz",
    BASEPATH / "fig1e_data/optimal_G20_run5.npz",
]

# Colors
DARK_BLUE     = '#00008B'
RED_COLOR     = '#D62728'
PAYLOAD_COLOR = '#737BA8'
GOAL_COLOR    = '#008F25'   # dark green

# Payload texture: Perlin noise mapped from light to dark grey
PAYLOAD_GRAY_LIGHT  = (0.86, 0.86, 0.86)
PAYLOAD_GRAY_DARK   = (0.20, 0.20, 0.20)
PAYLOAD_NOISE_CELLS = 3      # base noise cells across the payload diameter
PAYLOAD_NOISE_OCTAVES = 3
PAYLOAD_NOISE_SEED  = 7

PARTICLE_RADIUS_SCALE = 1.5   # display-size multiplier for particle circles (main view + inset)
# Particle score colouring: MINCOLOR_SCORE -> yellow, MAXCOLOR_SCORE -> purple.
# Scores outside [MIN, MAX] (incl. disconnected particles) clip to the end colours.
MINCOLOR_SCORE  = 0
MAXCOLOR_SCORE  = 50
SCORE_CMAP_LOW  = '#F4D03F'   # warm amber  — MINCOLOR_SCORE
SCORE_CMAP_HIGH = "#50106B"   # deep purple — MAXCOLOR_SCORE and above


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


def perlin_noise(size, cells, octaves=1, seed=0):
    """Fractal 2D Perlin noise, (size, size) array normalised to [0, 1]."""
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = cells * 2**o
        ang = rng.uniform(0, 2 * np.pi, (n + 1, n + 1))
        gx, gy = np.cos(ang), np.sin(ang)
        t = np.linspace(0, n, size, endpoint=False)
        xi = t.astype(int)
        f = t - xi
        u = f * f * f * (f * (f * 6 - 15) + 10)           # smootherstep fade
        X, Y = np.meshgrid(f, f)                           # X varies along columns
        U, V = np.meshgrid(u, u)
        ix, iy = np.meshgrid(xi, xi)

        def dot(dx, dy):
            return gx[iy + dy, ix + dx] * (X - dx) + gy[iy + dy, ix + dx] * (Y - dy)

        n00, n10, n01, n11 = dot(0, 0), dot(1, 0), dot(0, 1), dot(1, 1)
        val = (n00 * (1 - U) + n10 * U) * (1 - V) + (n01 * (1 - U) + n11 * U) * V
        out += amp * val
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / (out.max() - out.min())


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
            import figures.fig1e.optimal_path as _op
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
            ax.plot(xs, ys, color='#282828', linewidth=wall_lw, solid_capstyle='round', zorder=10)

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
            score_norm = Normalize(vmin=MINCOLOR_SCORE, vmax=MAXCOLOR_SCORE, clip=True)
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
        display_radii = np.maximum(particle_radius, min_r) * PARTICLE_RADIUS_SCALE
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

        # 5. Payload marker — adapted run only, Perlin-noise grey texture
        if draw_optimal:
            cx, cy = float(optimal_avg_end[0]), float(optimal_avg_end[1])
            noise = perlin_noise(256, PAYLOAD_NOISE_CELLS, PAYLOAD_NOISE_OCTAVES,
                                 PAYLOAD_NOISE_SEED)
            lo = np.array(PAYLOAD_GRAY_LIGHT); hi = np.array(PAYLOAD_GRAY_DARK)
            tex = lo + (hi - lo) * noise[..., None]
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            im = ax.imshow(tex, extent=(cx - payload_radius, cx + payload_radius,
                                        cy - payload_radius, cy + payload_radius),
                           origin='lower', interpolation='bilinear', alpha=1.0, zorder=10.5)
            clip = mpatches.Circle((cx, cy), radius=payload_radius, transform=ax.transData)
            im.set_clip_path(clip)
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
            ax.add_patch(mpatches.Circle((cx, cy), radius=payload_radius,
                                         facecolor='none', edgecolor='#282828',
                                         linewidth=1.0, zorder=10.6))

        # Goal
        ax.plot(goal_position[0], goal_position[1],
                '*', color=GOAL_COLOR, markersize=18, markeredgewidth=0.8,
                markeredgecolor='#003A0E', zorder=11)

        # ── Fill figure exactly ───────────────────────────────────────────────
        ax.set_position([0, 0, 1, 1])

        # ── Legend — top-left corner ───────────────
        legend_handles = []
        if draw_default:
            legend_handles += [                
                mlines.Line2D([], [], color=DARK_BLUE, alpha=1.0,  linewidth=2.5,
                              label='Original path'),
            ]
        if draw_optimal:
            legend_handles += [
                mlines.Line2D([], [], color=RED_COLOR, alpha=1.0,  linewidth=2.5,
                              label='Adapted path'),
            ]
        leg = ax.legend(
            handles=legend_handles,
            loc='upper left',
            bbox_to_anchor=(0.07, 0.98),
            fontsize=18,
            frameon=True,
            edgecolor='#aaaaaa',
        )
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_alpha(1.0)
        leg.set_zorder(20)

        # ── Inset colorbar for particle score (inside the legend box) ─────────
        # The legend's own frame is replaced by one white box that also
        # covers the colorbar, so legend + colorbar read as a single panel.
        if COLOR_BY_SCORE:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            inv = fig.transFigure.inverted()
            lb = leg.get_window_extent(renderer).transformed(inv)
            fw, fh = fig.get_size_inches()
            pad_x = leg.borderpad * 18 / 72 / fw      # legend interior padding
            pad_y = leg.borderpad * 18 / 72 / fh
            cb_h = 0.018
            BOX_GAP = 0.007   # extra white space on all four sides of the legend + colorbar content
            cax = fig.add_axes([lb.x0 + pad_x, lb.y0 - 0.022 - cb_h,
                                lb.width - 2 * pad_x, cb_h])
            cax.set_zorder(25)
            sm = plt.cm.ScalarMappable(norm=score_norm, cmap=score_cmap)
            cb = fig.colorbar(sm, cax=cax, orientation='horizontal', alpha=0.7)
            _mid = (MINCOLOR_SCORE + MAXCOLOR_SCORE) / 2
            cb.set_ticks([MINCOLOR_SCORE, _mid, MAXCOLOR_SCORE])
            cb.set_ticklabels([f'{MINCOLOR_SCORE:g}', f'{_mid:g}', f'≥{MAXCOLOR_SCORE:g}'])
            cb.ax.tick_params(labelsize=14)
            cb.set_label('Particle score', fontsize=16)
            cb.outline.set_linewidth(0.8)
            tl = cb.ax.get_xticklabels()
            tl[0].set_ha('left')
            tl[-1].set_ha('right')

            # Extend a single white box from the legend top to below the colorbar label
            fig.canvas.draw()
            tb = cb.ax.get_tightbbox(renderer).transformed(inv)
            leg.set_frame_on(False)
            box = mpatches.FancyBboxPatch(
                (lb.x0 - BOX_GAP / (fw / fh), tb.y0 - pad_y - BOX_GAP),
                lb.width + 2 * BOX_GAP / (fw / fh),
                (lb.y1 + BOX_GAP) - (tb.y0 - pad_y - BOX_GAP),
                boxstyle='round,pad=0,rounding_size=0.004',
                transform=fig.transFigure, facecolor='white',
                edgecolor='#282828', linewidth=wall_lw, joinstyle='round', zorder=19, clip_on=False)
            ax.add_artist(box)

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
