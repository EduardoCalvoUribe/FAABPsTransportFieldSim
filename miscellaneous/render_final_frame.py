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
from matplotlib.collections import PatchCollection
import numpy as np


SOURCE_FILE      = "maze_G5_run1.npz"
OUTPUT_IMAGE     = "visualizations/maze_G5_run1.png"
COLOR_BY_SCORE   = False   # True = rainbow by score, False = curvity blue/red
DRAW_OPENED_WALL = False   # True = draw red line where shortcut wall was removed
DRAW_SCORE_EDGES = True    # True = draw one directed edge per particle (score graph)
VIEW_RANGE       = 90.0    # Particle observation range r; overridden by NPZ if stored

import sys as _sys
if len(_sys.argv) > 1:
    SOURCE_FILE = _sys.argv[1]
    _stem = os.path.splitext(os.path.basename(SOURCE_FILE))[0]
    OUTPUT_IMAGE = f"visualizations/{_stem}.png"
if "--no-marker" in _sys.argv:
    DRAW_OPENED_WALL = False
if "--no-edges" in _sys.argv:
    DRAW_SCORE_EDGES = False


# ── Score-graph edge helpers ──────────────────────────────────────────────────

def _los_blocked_batch(x1, y1, x2s, y2s, wx1, wy1, wx2, wy2):
    """Check whether the straight line from (x1, y1) to each of the
    target points (x2s[k], y2s[k]) is blocked by any wall segment.

    Args:
        x1, y1   – source point (scalars)
        x2s, y2s – arrays of target x/y coordinates, shape (T,)
        wx1/wy1/wx2/wy2 – wall endpoint arrays, shape (W,)

    Returns:
        blocked  – boolean array of shape (T,); True if that target is blocked
    """
    n_targets = len(x2s)
    n_walls   = len(wx1)

    if n_walls == 0 or n_targets == 0:
        return np.zeros(n_targets, dtype=bool)

    # Broadcast shapes: T = targets, W = walls
    dx_ab = (x2s - x1)[:, None]          # (T, 1)  source→target vector
    dy_ab = (y2s - y1)[:, None]

    dx_ac = wx1[None, :] - x1            # (1, W)  source→wall-start
    dy_ac = wy1[None, :] - y1
    dx_ad = wx2[None, :] - x1            # (1, W)  source→wall-end
    dy_ad = wy2[None, :] - y1

    # Cross products: do C and D lie on opposite sides of AB?
    d1 = dx_ab * dy_ac - dy_ab * dx_ac   # (T, W)
    d2 = dx_ab * dy_ad - dy_ab * dx_ad   # (T, W)
    cond1 = d1 * d2 < 0

    dx_cd = (wx2 - wx1)[None, :]         # (1, W)  wall vector
    dy_cd = (wy2 - wy1)[None, :]

    dx_ca = x1 - wx1[None, :]            # (1, W)  wall-start→source
    dy_ca = y1 - wy1[None, :]
    dx_cb = x2s[:, None] - wx1[None, :]  # (T, W)  wall-start→target
    dy_cb = y2s[:, None] - wy1[None, :]

    # Do A and B lie on opposite sides of CD?
    d3 = dx_cd * dy_ca - dy_cd * dx_ca   # (1, W)
    d4 = dx_cd * dy_cb - dy_cd * dx_cb   # (T, W)
    cond2 = d3 * d4 < 0

    return np.any(cond1 & cond2, axis=1)  # (T,) – True if any wall blocks


def compute_score_edges(positions, scores, goal, walls, box_size, view_range,
                        display_radii):
    """Compute one outgoing directed edge per particle for the score graph.

    Rules (matching the simulation algorithm):
      • If the goal is within `view_range` AND no wall blocks the LoS →
        edge points to the goal.
      • Otherwise → edge points to the in-range, LoS-unblocked neighbor
        with the lowest score (ties broken by proximity).
      • Particles with no visible neighbors get no edge.

    The returned arrow endpoint is placed at the *edge* of the target
    particle's display circle (so arrowheads land just outside the disk).

    Returns:
        from_xy     – list of (fx, fy) source positions
        to_xy       – list of (tx, ty) arrow tip positions
        is_goal     – list of bools; True when the edge targets the goal
    """
    n    = len(positions)
    r2   = view_range ** 2
    half = box_size / 2.0

    # Ensure display_radii is a 1-D array of length n
    dr = np.asarray(display_radii, dtype=float)
    if dr.ndim == 0:
        dr = np.full(n, float(dr))

    # Pre-extract wall endpoints for the vectorised LoS check
    if walls.ndim == 2 and walls.shape[0] > 0:
        wx1 = walls[:, 0].astype(float)
        wy1 = walls[:, 1].astype(float)
        wx2 = walls[:, 2].astype(float)
        wy2 = walls[:, 3].astype(float)
    else:
        wx1 = wy1 = wx2 = wy2 = np.zeros(0)

    from_xy    = []
    to_xy      = []
    is_goal_l  = []

    for i in range(n):
        xi, yi = float(positions[i, 0]), float(positions[i, 1])

        # ── 1. Goal check ──────────────────────────────────────────────────
        gdx = goal[0] - xi
        gdy = goal[1] - yi
        if gdx >  half: gdx -= box_size
        elif gdx < -half: gdx += box_size
        if gdy >  half: gdy -= box_size
        elif gdy < -half: gdy += box_size

        if gdx * gdx + gdy * gdy <= r2:
            gx_near, gy_near = xi + gdx, yi + gdy
            blocked_g = _los_blocked_batch(
                xi, yi,
                np.array([gx_near]), np.array([gy_near]),
                wx1, wy1, wx2, wy2,
            )
            if not blocked_g[0]:
                from_xy.append((xi, yi))
                to_xy.append((gx_near, gy_near))
                is_goal_l.append(True)
                continue

        # ── 2. Find best neighbor ──────────────────────────────────────────
        dxv = positions[:, 0] - xi
        dyv = positions[:, 1] - yi
        # Minimum-image convention
        dxv = np.where(dxv >  half, dxv - box_size,
              np.where(dxv < -half, dxv + box_size, dxv))
        dyv = np.where(dyv >  half, dyv - box_size,
              np.where(dyv < -half, dyv + box_size, dyv))
        dist2 = dxv * dxv + dyv * dyv

        mask = (dist2 <= r2) & (np.arange(n) != i)
        cand  = np.where(mask)[0]
        if len(cand) == 0:
            continue

        cand_x = xi + dxv[cand]
        cand_y = yi + dyv[cand]
        blocked = _los_blocked_batch(xi, yi, cand_x, cand_y,
                                     wx1, wy1, wx2, wy2)
        cand = cand[~blocked]
        if len(cand) == 0:
            continue

        cand_scores = scores[cand]
        min_s  = cand_scores.min()
        tied   = cand[cand_scores == min_s]
        best_j = tied[np.argmin(dist2[tied])]   # closest among tied min-scorers

        bdx, bdy  = float(dxv[best_j]), float(dyv[best_j])
        norm      = float(np.sqrt(dist2[best_j]))
        rj        = float(dr[best_j])
        # Arrow tip placed at edge of target circle
        scale     = max(0.0, (norm - rj) / norm) if norm > rj else 1.0
        tx        = xi + bdx * scale
        ty        = yi + bdy * scale

        from_xy.append((xi, yi))
        to_xy.append((tx, ty))
        is_goal_l.append(False)

    return from_xy, to_xy, is_goal_l


# ── Colour helpers ────────────────────────────────────────────────────────────

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


def score_color(s):
    hue = (0.75 + (float(s) % 50) / 50.0) % 1.0
    return colorsys.hsv_to_rgb(hue, 1.0, 1.0)


PAYLOAD_COLOR  = '#737BA8'
GOAL_COLOR     = '#008F25'   # dark green
PATH_COLOR     = '#D62728'   # red

# Per-maze minimum visual particle size (set after box_size is known, inside __main__)
# G10/G20 → G5 reference (box/300);  G30 → G10 reference (box/600)


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

    # Use stored view range if available; otherwise fall back to the script constant
    if 'particle_view_range' in data:
        VIEW_RANGE = float(data['particle_view_range'])
        print(f"  view_range loaded from NPZ: {VIEW_RANGE}")

    # Opened wall segment (green dashed line showing the removed wall)
    # Loaded from NPZ when available; re-derived for older files that predate this field.
    opened_wall_seg = None
    if 'opened_wall_segment' in data:
        ows = np.array(data['opened_wall_segment'])
        if np.any(ows != 0):           # zeros → no shortcut was recorded
            opened_wall_seg = ows
    else:
        # Fallback: re-derive from grid geometry + seed=42 (matches main.py default)
        try:
            import adaptable_fig.optimal_path as _op
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

    print(f"  {n_particles} particles, {payload_trajectory.shape[0]} trajectory frames  ({time.time()-t0:.1f}s)")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_aspect('equal')
    ax.axis('off')

    # Walls (thinner for G30 where the maze is densest)
    wall_lw = 1.3 if box_size > 1200 else 1.8
    for i in range(walls.shape[0]):
        xs, ys = _arc_points(walls[i,0], walls[i,1], walls[i,2], walls[i,3], walls[i,4])
        ax.plot(xs, ys, color='#505050', linewidth=wall_lw, solid_capstyle='round', zorder=10)

    # Opened wall — red solid line where the shortcut wall was removed
    if DRAW_OPENED_WALL and opened_wall_seg is not None:
        ax.plot([opened_wall_seg[0], opened_wall_seg[2]],
                [opened_wall_seg[1], opened_wall_seg[3]],
                color='red', linewidth=2.5, linestyle='-',
                solid_capstyle='round', zorder=11)

    # Payload trajectory — solid line, drawn over payload (zorder=8 > payload zorder=7)
    ax.plot(payload_trajectory[:, 0], payload_trajectory[:, 1],
            color=PATH_COLOR, linewidth=3.0, linestyle='-', alpha=0.75, zorder=8)

    # Particles (final frame)
    if COLOR_BY_SCORE:
        colors = [score_color(s) for s in last_scores]
    else:
        colors = [curvity_color(c) for c in last_curvity]

    r = float(particle_radius[0]) if particle_radius.ndim > 0 else float(particle_radius)
    # Per-maze minimum display radius:
    #   G30  (box>1200) → G10 reference (box/600 = 3.0)
    #   G20  (box>600)  → slightly smaller than G5 (box/400 = 3.0)
    #   G10 and below   → G5 reference (box/300 = 2.0 for G10)
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

    # Payload (final position) — solid gold disk
    px_final, py_final = payload_trajectory[-1, 0], payload_trajectory[-1, 1]
    ax.add_patch(mpatches.Circle(
        (px_final, py_final), radius=payload_radius,
        facecolor=PAYLOAD_COLOR, edgecolor='none',
        alpha=1.0, zorder=7,
    ))

    # Goal
    ax.plot(goal_position[0], goal_position[1],
            '*', color=GOAL_COLOR, markersize=18, markeredgewidth=0.8, markeredgecolor='#003A0E', zorder=11)

    # ── Score-graph directed edges ────────────────────────────────────────
    if DRAW_SCORE_EDGES:
        print(f"  Computing score edges (N={n_particles}, r={VIEW_RANGE}, "
              f"W={walls.shape[0]}) ...")
        t_edge = time.time()
        from_xy, to_xy, is_goal_flags = compute_score_edges(
            last_positions, last_scores, goal_position, walls,
            box_size, VIEW_RANGE, display_radii,
        )
        print(f"  {len(from_xy)} edges computed  ({time.time()-t_edge:.1f}s)")

        # Separate into normal and goal edges for layered drawing
        norm_from, norm_to = [], []
        goal_from, goal_to = [], []
        for fxy, txy, ig in zip(from_xy, to_xy, is_goal_flags):
            if ig:
                goal_from.append(fxy)
                goal_to.append(txy)
            else:
                norm_from.append(fxy)
                norm_to.append(txy)

        # Scale arrowhead by maze size so it looks consistent across G2–G30
        ms_norm = max(4, min(10, box_size / 80.0))
        ms_goal = max(6, min(14, box_size / 60.0))

        # Draw normal (neighbour) edges — thin dark arrows, semi-transparent
        # zorder=12: above walls (10) and particles (6) so arrowheads are visible
        for (fx, fy), (tx, ty) in zip(norm_from, norm_to):
            ap = mpatches.FancyArrowPatch(
                (fx, fy), (tx, ty),
                arrowstyle='-|>',
                mutation_scale=ms_norm,
                color='#222222',
                linewidth=0.55,
                alpha=0.30,
                zorder=12,
            )
            ax.add_patch(ap)

        # Draw goal edges — slightly bolder green arrows on top
        for (fx, fy), (tx, ty) in zip(goal_from, goal_to):
            ap = mpatches.FancyArrowPatch(
                (fx, fy), (tx, ty),
                arrowstyle='-|>',
                mutation_scale=ms_goal,
                color=GOAL_COLOR,
                linewidth=1.1,
                alpha=0.70,
                zorder=13,
            )
            ax.add_patch(ap)

    # ── Fill figure exactly, then place inset ────────────────────────────
    ax.set_position([0, 0, 1, 1])

    # ── Inset: zoomed payload view — only for G10+ (box_size > 300) ──────
    if box_size > 300:
        # Use fig.add_axes (top-level) so the inset is fully opaque over the main axes.
        zoom_half = 40
        _ap = ax.get_position()   # read position AFTER tight_layout
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

        # Particle positions relative to final payload centre — no periodic wrapping
        rel_x = last_positions[:, 0] - px_final
        rel_y = last_positions[:, 1] - py_final
        inset_circles = [mpatches.Circle((dx, dy), radius=ri)
                         for dx, dy, ri in zip(rel_x, rel_y, particle_radius)]
        inset_pc = PatchCollection(inset_circles, facecolors=colors, alpha=0.7, linewidths=0, zorder=6)
        inset_ax.add_collection(inset_pc)

        # Particle ID labels — shown only for particles inside the visible zoom window
        label_offset = r * 1.4   # nudge label to the left of each particle (true radius)
        for i in range(n_particles):
            if abs(rel_x[i]) < zoom_half and abs(rel_y[i]) < zoom_half:
                inset_ax.text(rel_x[i] - label_offset, rel_y[i],
                              str(i), fontsize=6, ha='right', va='center',
                              color='black', zorder=9)

        # Payload at origin — solid gold disk
        inset_ax.add_patch(mpatches.Circle(
            (0, 0), radius=payload_radius,
            facecolor=PAYLOAD_COLOR, edgecolor='none',
            alpha=1.0, zorder=7,
        ))

        # Goal in payload-relative coordinates — no periodic wrapping
        gx_rel = goal_position[0] - px_final
        gy_rel = goal_position[1] - py_final
        inset_ax.plot(gx_rel, gy_rel, '*', color=GOAL_COLOR, markersize=24,
                      markeredgewidth=0.8, markeredgecolor='#003A0E', zorder=100)

        # Walls in payload-relative coordinates — no periodic wrapping
        for i in range(walls.shape[0]):
            wx1 = walls[i, 0] - px_final
            wy1 = walls[i, 1] - py_final
            wx2 = walls[i, 2] - px_final
            wy2 = walls[i, 3] - py_final
            xs, ys = _arc_points(wx1, wy1, wx2, wy2, walls[i, 4])
            inset_ax.plot(xs, ys, color='#505050', linewidth=1.5, solid_capstyle='round', zorder=10)

        # Opened wall in payload-relative coordinates
        if DRAW_OPENED_WALL and opened_wall_seg is not None:
            inset_ax.plot(
                [opened_wall_seg[0] - px_final, opened_wall_seg[2] - px_final],
                [opened_wall_seg[1] - py_final, opened_wall_seg[3] - py_final],
                color='red', linewidth=2.0, linestyle='-',
                solid_capstyle='round', zorder=11)

        # Payload trajectory relative to final payload position — no periodic wrapping
        traj_x = payload_trajectory[:, 0] - px_final
        traj_y = payload_trajectory[:, 1] - py_final
        inset_ax.plot(traj_x, traj_y, color=PATH_COLOR, linewidth=3.0, linestyle='-', alpha=0.75, zorder=8)

    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Saved to {OUTPUT_IMAGE}  (total: {time.time()-t0:.1f}s)")
