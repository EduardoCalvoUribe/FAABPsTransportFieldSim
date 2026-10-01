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
OUTPUT_IMAGE     = "graph.png"           # bare filename → saves in cwd (fig3c/)
COLOR_BY_SCORE   = False   # True = rainbow by score, False = curvity blue/red
DRAW_OPENED_WALL = False   # True = draw red line where shortcut wall was removed
DRAW_SCORE_EDGES = True    # True = draw one directed edge per particle (score graph)
SHOW_PAYLOAD_PATH = False  # True = draw payload trajectory line
VIEW_RANGE       = 90.0    # Particle observation range r; overridden by NPZ if stored

# Curvity piecewise-linear parameters — must match the run that produced the NPZ
MAX_CURVITY =  0.5
MIN_CURVITY = -1.0
MID_CURVITY = -0.25

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

    Returns a boolean array of shape (T,); True means blocked.
    """
    n_targets = len(x2s)
    n_walls   = len(wx1)
    if n_walls == 0 or n_targets == 0:
        return np.zeros(n_targets, dtype=bool)

    dx_ab = (x2s - x1)[:, None]
    dy_ab = (y2s - y1)[:, None]

    dx_ac = wx1[None, :] - x1
    dy_ac = wy1[None, :] - y1
    dx_ad = wx2[None, :] - x1
    dy_ad = wy2[None, :] - y1

    d1 = dx_ab * dy_ac - dy_ab * dx_ac
    d2 = dx_ab * dy_ad - dy_ab * dx_ad
    cond1 = d1 * d2 < 0

    dx_cd = (wx2 - wx1)[None, :]
    dy_cd = (wy2 - wy1)[None, :]
    dx_ca = x1 - wx1[None, :]
    dy_ca = y1 - wy1[None, :]
    dx_cb = x2s[:, None] - wx1[None, :]
    dy_cb = y2s[:, None] - wy1[None, :]

    d3 = dx_cd * dy_ca - dy_cd * dx_ca
    d4 = dx_cd * dy_cb - dy_cd * dx_cb
    cond2 = d3 * d4 < 0

    return np.any(cond1 & cond2, axis=1)


def compute_score_edges(positions, scores, goal, walls, box_size, view_range,
                        display_radii):
    """Compute one outgoing directed edge per particle for the score graph.

    Rules (matching the simulation algorithm):
      • Goal within view_range with unobstructed LoS → edge to goal.
      • Otherwise → edge to in-range, LoS-visible neighbor with lowest score
        (ties broken by proximity).
      • Particles with no visible neighbors get no edge.

    Arrow tips are placed at the edge of the target particle's display circle.
    Returns (from_xy, to_xy, is_goal) lists.
    """
    n    = len(positions)
    r2   = view_range ** 2
    half = box_size / 2.0

    dr = np.asarray(display_radii, dtype=float)
    if dr.ndim == 0:
        dr = np.full(n, float(dr))

    if walls.ndim == 2 and walls.shape[0] > 0:
        wx1 = walls[:, 0].astype(float)
        wy1 = walls[:, 1].astype(float)
        wx2 = walls[:, 2].astype(float)
        wy2 = walls[:, 3].astype(float)
    else:
        wx1 = wy1 = wx2 = wy2 = np.zeros(0)

    from_xy, to_xy, is_goal_l = [], [], []

    for i in range(n):
        xi, yi = float(positions[i, 0]), float(positions[i, 1])

        # ── goal check ────────────────────────────────────────────────────
        gdx = goal[0] - xi
        gdy = goal[1] - yi
        if gdx >  half: gdx -= box_size
        elif gdx < -half: gdx += box_size
        if gdy >  half: gdy -= box_size
        elif gdy < -half: gdy += box_size

        if gdx * gdx + gdy * gdy <= r2:
            gx_n, gy_n = xi + gdx, yi + gdy
            if not _los_blocked_batch(xi, yi,
                                      np.array([gx_n]), np.array([gy_n]),
                                      wx1, wy1, wx2, wy2)[0]:
                from_xy.append((xi, yi))
                to_xy.append((gx_n, gy_n))
                is_goal_l.append(True)
                continue

        # ── neighbor search ───────────────────────────────────────────────
        dxv = positions[:, 0] - xi
        dyv = positions[:, 1] - yi
        dxv = np.where(dxv >  half, dxv - box_size,
              np.where(dxv < -half, dxv + box_size, dxv))
        dyv = np.where(dyv >  half, dyv - box_size,
              np.where(dyv < -half, dyv + box_size, dyv))
        dist2 = dxv * dxv + dyv * dyv

        cand = np.where((dist2 <= r2) & (np.arange(n) != i))[0]
        if len(cand) == 0:
            continue

        blocked = _los_blocked_batch(xi, yi,
                                     xi + dxv[cand], yi + dyv[cand],
                                     wx1, wy1, wx2, wy2)
        cand = cand[~blocked]
        if len(cand) == 0:
            continue

        cand_scores = scores[cand]
        min_s  = cand_scores.min()
        tied   = cand[cand_scores == min_s]
        best_j = tied[np.argmin(dist2[tied])]

        bdx, bdy = float(dxv[best_j]), float(dyv[best_j])
        norm     = float(np.sqrt(dist2[best_j]))
        rj       = float(dr[best_j]) * 2.5
        scale    = max(0.0, (norm - rj) / norm) if norm > rj else 1.0

        from_xy.append((xi, yi))
        to_xy.append((xi + bdx * scale, yi + bdy * scale))
        is_goal_l.append(False)

    return from_xy, to_xy, is_goal_l


def compute_polarity_vectors(positions, scores, goal, walls, box_size, view_range):
    """Return (N, 2) unit polarity vectors — same neighbour logic as compute_score_edges
    but pointing to the exact target centre rather than the display-circle edge."""
    n    = len(positions)
    r2   = view_range ** 2
    half = box_size / 2.0
    polarity = np.zeros((n, 2), dtype=float)

    if walls.ndim == 2 and walls.shape[0] > 0:
        wx1 = walls[:, 0].astype(float); wy1 = walls[:, 1].astype(float)
        wx2 = walls[:, 2].astype(float); wy2 = walls[:, 3].astype(float)
    else:
        wx1 = wy1 = wx2 = wy2 = np.zeros(0)

    for i in range(n):
        xi, yi = float(positions[i, 0]), float(positions[i, 1])

        gdx = goal[0] - xi;  gdy = goal[1] - yi
        if gdx >  half: gdx -= box_size
        elif gdx < -half: gdx += box_size
        if gdy >  half: gdy -= box_size
        elif gdy < -half: gdy += box_size

        if gdx * gdx + gdy * gdy <= r2:
            gx_n, gy_n = xi + gdx, yi + gdy
            if not _los_blocked_batch(xi, yi, np.array([gx_n]), np.array([gy_n]),
                                      wx1, wy1, wx2, wy2)[0]:
                nm = np.sqrt(gdx * gdx + gdy * gdy)
                if nm > 1e-10:
                    polarity[i] = [gdx / nm, gdy / nm]
                continue

        dxv = positions[:, 0] - xi;  dyv = positions[:, 1] - yi
        dxv = np.where(dxv >  half, dxv - box_size, np.where(dxv < -half, dxv + box_size, dxv))
        dyv = np.where(dyv >  half, dyv - box_size, np.where(dyv < -half, dyv + box_size, dyv))
        dist2 = dxv * dxv + dyv * dyv

        cand = np.where((dist2 <= r2) & (np.arange(n) != i))[0]
        if len(cand) == 0:
            continue
        blocked = _los_blocked_batch(xi, yi, xi + dxv[cand], yi + dyv[cand],
                                     wx1, wy1, wx2, wy2)
        cand = cand[~blocked]
        if len(cand) == 0:
            continue

        cand_scores = scores[cand]
        min_s = cand_scores.min()
        tied  = cand[cand_scores == min_s]
        best_j = tied[np.argmin(dist2[tied])]

        bdx, bdy = float(dxv[best_j]), float(dyv[best_j])
        nm = np.sqrt(bdx * bdx + bdy * bdy)
        if nm > 1e-10:
            polarity[i] = [bdx / nm, bdy / nm]

    return polarity


# ── Arc / colour helpers ──────────────────────────────────────────────────────

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
    _out_dir = os.path.dirname(OUTPUT_IMAGE)
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)

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

    print(f"  {n_particles} particles, {payload_trajectory.shape[0]} trajectory frames  ({time.time()-t0:.1f}s)")

    # ── Compute & save polarity + heading vectors ─────────────────────────
    print("  Computing polarity vectors ...")
    polarity_vecs = compute_polarity_vectors(
        last_positions, last_scores, goal_position, walls, box_size, VIEW_RANGE,
    )

    # Invert piecewise-linear map: curvity → dot = orientation · polarity
    c = last_curvity.ravel()
    dot = np.where(
        c <= MID_CURVITY,
        (c - MID_CURVITY) / (MIN_CURVITY - MID_CURVITY),
        (c - MAX_CURVITY) / (MID_CURVITY - MAX_CURVITY) - 1.0,
    )
    dot = np.clip(dot, -1.0, 1.0)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - dot ** 2))
    signs = np.random.choice([-1.0, 1.0], size=n_particles)
    perp = np.column_stack([-polarity_vecs[:, 1], polarity_vecs[:, 0]])  # (-py, px)
    heading_vecs = dot[:, None] * polarity_vecs + (signs * sin_theta)[:, None] * perp
    hn = np.linalg.norm(heading_vecs, axis=1, keepdims=True)
    safe_hn = np.where(hn > 1e-10, hn, 1.0)
    heading_vecs = np.where(hn > 1e-10, heading_vecs / safe_hn, heading_vecs)

    # For particles touching/near the payload: override random sign with the heading
    # that most closely points toward the goal.
    px_final, py_final = payload_trajectory[-1, 0], payload_trajectory[-1, 1]
    near_thresh = payload_radius + float(particle_radius[0]) * 1.5
    to_goal = goal_position - last_positions          # (N, 2), unnormalised
    for i in range(n_particles):
        dx = last_positions[i, 0] - px_final
        dy = last_positions[i, 1] - py_final
        if dx * dx + dy * dy <= near_thresh ** 2:
            h_pos = dot[i] * polarity_vecs[i] + sin_theta[i] * perp[i]
            h_neg = dot[i] * polarity_vecs[i] - sin_theta[i] * perp[i]
            if np.dot(h_pos, to_goal[i]) >= np.dot(h_neg, to_goal[i]):
                heading_vecs[i] = h_pos
            else:
                heading_vecs[i] = h_neg
            nm = np.linalg.norm(heading_vecs[i])
            if nm > 1e-10:
                heading_vecs[i] /= nm

    # Re-save NPZ with the two new arrays appended
    existing = {k: np.array(data[k]) for k in data.keys()}
    existing['polarity'] = polarity_vecs
    existing['orientations'] = heading_vecs
    _save_stem = SOURCE_FILE[:-4] if SOURCE_FILE.endswith('.npz') else SOURCE_FILE
    np.savez(_save_stem, **existing)
    print(f"  Saved polarity + orientations -> {SOURCE_FILE}")

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
        ax.plot(xs, ys, color='#282828', linewidth=wall_lw, solid_capstyle='round', zorder=10)

    # Opened wall — red solid line where the shortcut wall was removed
    if DRAW_OPENED_WALL and opened_wall_seg is not None:
        ax.plot([opened_wall_seg[0], opened_wall_seg[2]],
                [opened_wall_seg[1], opened_wall_seg[3]],
                color='red', linewidth=2.5, linestyle='-',
                solid_capstyle='round', zorder=11)

    # Payload trajectory — solid line, drawn over payload (zorder=8 > payload zorder=7)
    if SHOW_PAYLOAD_PATH:
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
    from_xy, to_xy, is_goal_flags = [], [], []
    if DRAW_SCORE_EDGES:
        print(f"  Computing score edges (N={n_particles}, r={VIEW_RANGE}, "
              f"W={walls.shape[0]}) ...")
        t_edge = time.time()
        from_xy, to_xy, is_goal_flags = compute_score_edges(
            last_positions, last_scores, goal_position, walls,
            box_size, VIEW_RANGE, display_radii,
        )
        print(f"  {len(from_xy)} edges computed  ({time.time()-t_edge:.1f}s)")

        norm_from, norm_to = [], []
        goal_from, goal_to = [], []
        for fxy, txy, ig in zip(from_xy, to_xy, is_goal_flags):
            if ig:
                goal_from.append(fxy)
                goal_to.append(txy)
            else:
                norm_from.append(fxy)
                norm_to.append(txy)

        ms_norm = max(6, min(15, box_size / 55.0)) * 1.3
        ms_goal = max(8, min(18, box_size / 45.0)) * 1.3

        # Normal (neighbour) edges — thin dark arrows, zorder=12 (above walls)
        for (fx, fy), (tx, ty) in zip(norm_from, norm_to):
            ax.add_patch(mpatches.FancyArrowPatch(
                (fx, fy), (tx, ty),
                arrowstyle='-|>', mutation_scale=ms_norm,
                color='#222222', linewidth=0.7, alpha=0.6, zorder=12,
            ))

        # Goal edges — bolder arrows, same gray as normal edges
        for (fx, fy), (tx, ty) in zip(goal_from, goal_to):
            ax.add_patch(mpatches.FancyArrowPatch(
                (fx, fy), (tx, ty),
                arrowstyle='-|>', mutation_scale=ms_goal,
                color='#222222', linewidth=0.7, alpha=0.6, zorder=13,
            ))

    # ── Fill figure exactly, then place inset ────────────────────────────
    ax.set_position([0, 0, 1, 1])

    # ── Inset: zoom of square [150,220] x [150,220], bottom-right ────────
    INSET_X0, INSET_X1 = 150.0, 220.0
    INSET_Y0, INSET_Y1 = 150.0, 220.0
    _ap = ax.get_position()
    _inset_w = 0.43 * _ap.width
    _inset_h = 0.43 * _ap.height
    inset_ax = fig.add_axes([
        _ap.x0 + _ap.width - 0.02 * _ap.width - _inset_w,
        _ap.y0 + 0.02 * _ap.height,
        _inset_w,
        _inset_h,
    ])
    inset_ax.set_xlim(INSET_X0, INSET_X1)
    inset_ax.set_ylim(INSET_Y0, INSET_Y1)
    inset_ax.set_aspect('equal')
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_linewidth(2)
    inset_ax.set_facecolor('white')

    # Faint box around the target area in the main axes
    ax.add_patch(mpatches.Rectangle(
        (INSET_X0, INSET_Y0), INSET_X1 - INSET_X0, INSET_Y1 - INSET_Y0,
        linewidth=1.0, edgecolor='#888888', facecolor='#AAAAAA', alpha=0.28,
        linestyle='--', zorder=15,
    ))

    # Connector lines from inset corners to corresponding points in main axes
    for _xy in [(INSET_X0, INSET_Y0), (INSET_X1, INSET_Y1)]:
        fig.add_artist(mpatches.ConnectionPatch(
            xyA=_xy, coordsA='data', axesA=inset_ax,
            xyB=_xy, coordsB='data', axesB=ax,
            color='#888888', linewidth=0.8, alpha=0.4, linestyle='--', zorder=20,
        ))

    # Particles in the inset region
    in_idx = [i for i in range(n_particles)
              if INSET_X0 - float(display_radii[i]) <= last_positions[i, 0] <= INSET_X1 + float(display_radii[i])
              and INSET_Y0 - float(display_radii[i]) <= last_positions[i, 1] <= INSET_Y1 + float(display_radii[i])]
    if in_idx:
        inset_circles = [mpatches.Circle((last_positions[i, 0], last_positions[i, 1]),
                                         radius=float(display_radii[i])) for i in in_idx]
        inset_pc = PatchCollection(inset_circles, facecolors=[colors[i] for i in in_idx],
                                   alpha=0.7, linewidths=0, zorder=6)
        inset_ax.add_collection(inset_pc)

    # Walls — draw all; axes clipping handles the rest
    for i in range(walls.shape[0]):
        xs, ys = _arc_points(walls[i, 0], walls[i, 1], walls[i, 2], walls[i, 3], walls[i, 4])
        inset_ax.plot(xs, ys, color='#282828', linewidth=1.5, solid_capstyle='round', zorder=10)

    # Goal if inside the inset region
    if INSET_X0 <= goal_position[0] <= INSET_X1 and INSET_Y0 <= goal_position[1] <= INSET_Y1:
        inset_ax.plot(goal_position[0], goal_position[1],
                      '*', color=GOAL_COLOR, markersize=18,
                      markeredgewidth=0.8, markeredgecolor='#003A0E', zorder=11)

    # Score edges as fixed-length arrows (direction only, length = 10 data units)
    unit_len = 10.0
    for (fx, fy), (tx, ty), ig in zip(from_xy, to_xy, is_goal_flags):
        if not (INSET_X0 <= fx <= INSET_X1 and INSET_Y0 <= fy <= INSET_Y1):
            continue
        dx, dy = tx - fx, ty - fy
        d = np.sqrt(dx * dx + dy * dy)
        if d < 1e-10:
            continue
        ex, ey = fx + dx / d * unit_len, fy + dy / d * unit_len
        inset_ax.add_patch(mpatches.FancyArrowPatch(
            (fx, fy), (ex, ey),
            arrowstyle='-|>', mutation_scale=8,
            color=GOAL_COLOR if ig else '#222222',
            linewidth=0.7, alpha=0.8, zorder=12,
        ))

    # Shade the corridor between the target box and the inset.
    # Hexagon: target-box bottom-right face → connector2 → inset top-left face → connector1.
    # This polygon never overlaps the inset interior.
    fig.canvas.draw()
    _main_to_fig  = ax.transData       + fig.transFigure.inverted()
    _inset_to_fig = inset_ax.transData + fig.transFigure.inverted()
    _p_main_bl  = _main_to_fig.transform( [INSET_X0, INSET_Y0])
    _p_main_br  = _main_to_fig.transform( [INSET_X1, INSET_Y0])
    _p_main_tr  = _main_to_fig.transform( [INSET_X1, INSET_Y1])
    _p_inset_tr = _inset_to_fig.transform([INSET_X1, INSET_Y1])
    _p_inset_tl = _inset_to_fig.transform([INSET_X0, INSET_Y1])
    _p_inset_bl = _inset_to_fig.transform([INSET_X0, INSET_Y0])
    fig.add_artist(mpatches.Polygon(
        [_p_main_bl, _p_main_br, _p_main_tr,
         _p_inset_tr, _p_inset_tl, _p_inset_bl],
        facecolor='#AAAAAA', alpha=0.075, edgecolor='none',
        transform=fig.transFigure, zorder=14,
    ))

    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Saved to {OUTPUT_IMAGE}  (total: {time.time()-t0:.1f}s)")
