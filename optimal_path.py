"""
optimal_path.py — Create a maze with exactly two routes to the goal.

Takes a Wilson-generated perfect maze and removes the single wall that
produces the most clearly optimal vs. sub-optimal route pair, defined as
the wall whose removal saves the greatest number of steps (maximum shortcut
*improvement* = gain − 1 − branch_depth_A − branch_depth_B).

Theory
------
A perfect maze is a spanning tree of the grid graph.  Every cell X belongs
to exactly one "path segment": the portion of the main path P that its
sub-tree hangs off.  Define

    idx(X)   = index i of the path node c_i whose branch X belongs to
    depth(X) = number of hops from X up to c_i  (0 if X is on P itself)

Adding passage A↔B creates two distinct routes from start→goal iff
idx(A) ≠ idx(B).  The shortcut saves

    improvement = |idx(A) − idx(B)| − 1 − depth(A) − depth(B)  steps

We pick the wall with the largest positive improvement.

Library use (called by main.py)
────────────────────────────────
    import optimal_path

    passages, info = optimal_path.open_best_shortcut(
        grid_size=10, seed=42,
        start_cell=(0, 0), goal_cell=(9, 9),
        box_size=600.0,
        output_prefix="optimal_path_G10",
    )
    WALLS = maze_to_walls(passages, MAZE_GRID_SIZE, BOX_SIZE)

Standalone
──────────
    python optimal_path.py <grid_size> [seed] [output_prefix]
"""

from __future__ import annotations

from collections import deque
import copy
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from src import wilson


# ─────────────────────────────────────────────────────────────────────────────
# Core graph helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_main_path(passages, start, goal):
    """BFS shortest path from *start* to *goal* in *passages*.

    In a perfect maze (spanning tree) this is the unique path.

    Parameters
    ----------
    passages : dict mapping (r, c) → set of (r, c) neighbours
    start    : (row, col) tuple
    goal     : (row, col) tuple

    Returns
    -------
    list of (row, col) including both endpoints, or None if unreachable.
    """
    queue   = deque([(start, [start])])
    visited = {start}
    while queue:
        cell, path = queue.popleft()
        if cell == goal:
            return path
        for nb in passages.get(cell, set()):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return None


def _compute_path_index_and_depth(passages, path):
    """Assign every cell its path-index and branch depth via flood-fill.

    The main path P = [c_0, c_1, ..., c_k] divides the spanning tree into
    branches.  Each branch hangs off exactly one P-node c_i.

    Returns
    -------
    idx   : dict  cell → int   index i of the nearest path node
    depth : dict  cell → int   hops from cell up to that path node (0 on P)
    """
    idx   = {cell: i for i, cell in enumerate(path)}
    depth = {cell: 0 for cell in path}
    queue = deque(path)
    while queue:
        cell = queue.popleft()
        for nb in passages.get(cell, set()):
            if nb not in idx:
                idx[nb]   = idx[cell]
                depth[nb] = depth[cell] + 1
                queue.append(nb)
    return idx, depth


def find_shortcut_walls(passages, grid_size, start_cell, goal_cell):
    """Find every interior wall whose removal creates two routes where the
    shortcut is strictly shorter than the original.

    Parameters
    ----------
    passages  : perfect-maze passage dict
    grid_size : int, side length of the square grid
    start_cell, goal_cell : (row, col) tuples

    Returns
    -------
    candidates    : list of (improvement, gain, cell_A, cell_B),
                    sorted by improvement descending.
    original_path : list of cells — the unique path in the perfect maze.
    """
    original_path = find_main_path(passages, start_cell, goal_cell)
    if original_path is None:
        raise ValueError("No path from start to goal found in this maze.")

    idx, dep = _compute_path_index_and_depth(passages, original_path)

    candidates = []
    for r in range(grid_size):
        for c in range(grid_size):
            for dr, dc in [(0, 1), (1, 0)]:          # each interior wall once
                nr, nc = r + dr, c + dc
                if nr >= grid_size or nc >= grid_size:
                    continue
                A, B = (r, c), (nr, nc)
                if B in passages.get(A, set()):       # passage already exists
                    continue
                iA = idx.get(A)
                iB = idx.get(B)
                if iA is None or iB is None or iA == iB:
                    continue
                dA, dB      = dep.get(A, 0), dep.get(B, 0)
                gain        = abs(iA - iB)
                improvement = gain - 1 - dA - dB     # steps saved by shortcut
                if improvement > 0:
                    candidates.append((improvement, gain, A, B))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates, original_path


# ─────────────────────────────────────────────────────────────────────────────
# Main public API
# ─────────────────────────────────────────────────────────────────────────────

def open_best_shortcut(
    grid_size,
    seed,
    start_cell,
    goal_cell,
    box_size=None,
    output_prefix="optimal_path",
):
    """Generate a perfect maze, open the wall with the largest shortcut
    improvement, measure both resulting paths, and write analysis files.

    Writes *output_prefix*.txt  (path measurements)
    and    *output_prefix*.png  (maze with both paths drawn).

    Parameters
    ----------
    grid_size     : int   — side length of the square maze grid.
    seed          : int | None — RNG seed for Wilson's algorithm.
    start_cell    : (row, col) — starting cell (payload start).
    goal_cell     : (row, col) — goal cell.
    box_size      : float | None — simulation-unit box size (default 60×grid_size).
    output_prefix : str   — base filename (no extension) for output files.

    Returns
    -------
    passages : dict — modified passages with the shortcut opened.
    info     : dict — full path statistics (keys documented below).

    info keys
    ---------
    grid_size, seed, box_size, cell_size, start_cell, goal_cell
    opened_wall   : (cell_A, cell_B)
    gain_cells    : path-index gain (abs(idx_A - idx_B))
    improvement   : steps saved by the shortcut
    path1         : list of cells — original (longer) path
    path1_cells, path1_steps, path1_length
    path2         : list of cells — shortcut (shorter) path
    path2_cells, path2_steps, path2_length
    ratio         : path1_length / path2_length
    """
    if box_size is None:
        box_size = 60.0 * grid_size
    cell_size = box_size / grid_size

    # 1. Perfect maze
    base_passages = wilson.generate(grid_size, seed=seed)

    # 2. Best wall to remove
    candidates, original_path = find_shortcut_walls(
        base_passages, grid_size, start_cell, goal_cell)

    if not candidates:
        raise RuntimeError(
            "No wall found whose removal creates a strictly shorter shortcut. "
            "Try a different seed or a larger maze."
        )

    improvement, gain, cell_A, cell_B = candidates[0]

    # Compute the wall segment in simulation coordinates (used for rendering)
    (rA, cA_), (rB, cB_) = cell_A, cell_B
    if rA == rB:   # horizontal neighbours → the wall between them was vertical
        wx  = (min(cA_, cB_) + 1) * cell_size
        opened_wall_segment = np.array([wx, rA * cell_size, wx, (rA + 1) * cell_size])
    else:          # vertical neighbours → the wall between them was horizontal
        wy  = (min(rA, rB) + 1) * cell_size
        opened_wall_segment = np.array([cA_ * cell_size, wy, (cA_ + 1) * cell_size, wy])

    # 3. Open the passage
    passages = copy.deepcopy(base_passages)
    passages.setdefault(cell_A, set()).add(cell_B)
    passages.setdefault(cell_B, set()).add(cell_A)

    # 4. Measure both paths
    # Path 1: the unique path in the original perfect maze (still valid)
    path1  = original_path
    # Path 2: BFS on the modified maze finds the strictly shorter route
    path2  = find_main_path(passages, start_cell, goal_cell)

    steps1  = len(path1) - 1
    steps2  = len(path2) - 1
    length1 = steps1 * cell_size
    length2 = steps2 * cell_size
    ratio   = length1 / length2 if length2 > 0 else float('inf')

    info = dict(
        grid_size    = grid_size,
        seed         = seed,
        box_size     = box_size,
        cell_size    = cell_size,
        start_cell   = start_cell,
        goal_cell    = goal_cell,
        opened_wall          = (cell_A, cell_B),
        opened_wall_segment  = opened_wall_segment,   # [x1,y1,x2,y2] in sim units
        gain_cells           = gain,
        improvement          = improvement,
        path1        = path1,
        path1_cells  = len(path1),
        path1_steps  = steps1,
        path1_length = length1,
        path2        = path2,
        path2_cells  = len(path2),
        path2_steps  = steps2,
        path2_length = length2,
        ratio        = ratio,
    )

    # 5. Write outputs
    _write_text(info, f"{output_prefix}.txt")
    _draw_png(passages, info, f"{output_prefix}.png")

    return passages, info


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_text(info, filepath):
    gs   = info['grid_size'];    sd   = info['seed']
    bs   = info['box_size'];     cs   = info['cell_size']
    sc   = info['start_cell'];   gc   = info['goal_cell']
    cA, cB = info['opened_wall']
    gain = info['gain_cells'];   imp  = info['improvement']
    p1c  = info['path1_cells'];  p1s  = info['path1_steps'];  p1l = info['path1_length']
    p2c  = info['path2_cells'];  p2s  = info['path2_steps'];  p2l = info['path2_length']
    ratio = info['ratio']

    step_word = 'step' if imp == 1 else 'steps'
    lines = [
        "=== Optimal Path Analysis ===",
        f"Maze        : {gs}×{gs} grid,  seed = {sd}",
        f"Box size    : {bs:.1f} simulation units  (cell size = {cs:.1f})",
        f"Start cell  : {sc}",
        f"Goal cell   : {gc}",
        "",
        f"Opened wall : cell {cA} ↔ cell {cB}",
        f"              path-index gain = {gain} cells,  "
            f"shortcut saves {imp} {step_word}",
        "",
        "─" * 72,
        "Path 1  (original / longer):",
        f"  Cells  : {p1c}",
        f"  Steps  : {p1s}",
        f"  Length : {p1l:.2f} simulation units",
        "",
        "Path 2  (shortcut / shorter):",
        f"  Cells  : {p2c}",
        f"  Steps  : {p2s}",
        f"  Length : {p2l:.2f} simulation units",
        "",
        f"Ratio   : {ratio:.4f}×  "
            f"(path 1 is {(ratio - 1) * 100:.1f}% longer than path 2)",
        "─" * 72,
    ]
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[optimal_path] Measurements written to  : {filepath}")


def _draw_png(passages, info, filepath):
    gs  = info['grid_size'];    bs  = info['box_size'];    cs  = info['cell_size']
    sc  = info['start_cell'];   gc  = info['goal_cell']
    cA, cB = info['opened_wall']
    path1  = info['path1'];     path2 = info['path2']

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, bs);  ax.set_ylim(0, bs)
    ax.set_aspect('equal');  ax.axis('off')
    ax.set_title(
        f"Maze {gs}×{gs}  –  two routes to goal\n"
        f"Path 1 (red): {info['path1_steps']} steps  |  "
        f"Path 2 (blue): {info['path2_steps']} steps  |  "
        f"ratio {info['ratio']:.2f}×",
        fontsize=9,
    )

    # ── Maze walls from the modified passages ──────────────────────────────
    # The opened wall is already absent from passages, so it won't be drawn.
    for r in range(gs - 1):
        for c in range(gs):
            if (r + 1, c) not in passages.get((r, c), set()):
                y = (r + 1) * cs
                ax.plot([c * cs, (c + 1) * cs], [y, y],
                        color='black', lw=1.5, solid_capstyle='round')
    for r in range(gs):
        for c in range(gs - 1):
            if (r, c + 1) not in passages.get((r, c), set()):
                x = (c + 1) * cs
                ax.plot([x, x], [r * cs, (r + 1) * cs],
                        color='black', lw=1.5, solid_capstyle='round')
    # Boundary box
    ax.plot([0, bs, bs, 0, 0], [0, 0, bs, bs, 0], color='black', lw=2.0)

    # ── Opened wall: dashed green line where the wall used to be ──────────
    (rA, cA_), (rB, cB_) = cA, cB
    if rA == rB:                               # horizontal neighbours → vertical wall
        wx   = (min(cA_, cB_) + 1) * cs
        wy1  = rA * cs;  wy2 = (rA + 1) * cs
        ax.plot([wx, wx], [wy1, wy2],
                color='limegreen', lw=2.5, ls='--', zorder=8,
                label='opened wall (shortcut)')
    else:                                      # vertical neighbours → horizontal wall
        wy   = (min(rA, rB) + 1) * cs
        wx1  = cA_ * cs;  wx2 = (cA_ + 1) * cs
        ax.plot([wx1, wx2], [wy, wy],
                color='limegreen', lw=2.5, ls='--', zorder=8,
                label='opened wall (shortcut)')

    # ── Draw paths through cell centres ───────────────────────────────────
    def cells_to_xy(path):
        xs = [(c + 0.5) * cs for _, c in path]
        ys = [(r + 0.5) * cs for r, _ in path]
        return xs, ys

    x1, y1 = cells_to_xy(path1)
    x2, y2 = cells_to_xy(path2)

    ax.plot(x1, y1, color='red', lw=2.0, alpha=0.85, zorder=7,
            label=f"Path 1 (original):  {info['path1_steps']} steps "
                  f"= {info['path1_length']:.0f} units")
    ax.plot(x2, y2, color='royalblue', lw=2.0, alpha=0.85, zorder=7,
            label=f"Path 2 (shortcut):  {info['path2_steps']} steps "
                  f"= {info['path2_length']:.0f} units")

    # ── Payload start (grey circle) and goal (green star) ─────────────────
    pr = max(cs * 0.18, 1.5)
    ax.add_patch(Circle(
        ((sc[1] + 0.5) * cs, (sc[0] + 0.5) * cs),
        pr, color='dimgray', alpha=0.85, zorder=9))
    ax.plot(
        (gc[1] + 0.5) * cs, (gc[0] + 0.5) * cs,
        'g*', markersize=14, markeredgewidth=1.2,
        markeredgecolor='darkgreen', zorder=9)

    ax.legend(loc='upper left', fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[optimal_path] Maze image saved to       : {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _grid   = int(sys.argv[1])  if len(sys.argv) > 1 else 10
    _seed   = int(sys.argv[2])  if len(sys.argv) > 2 else 42
    _prefix = sys.argv[3]       if len(sys.argv) > 3 else \
              f"optimal_path_G{_grid}_seed{_seed}"

    _start = (0, 0)
    _goal  = (_grid - 1, _grid - 1)

    _, _info = open_best_shortcut(
        grid_size   = _grid,
        seed        = _seed,
        start_cell  = _start,
        goal_cell   = _goal,
        output_prefix = _prefix,
    )

    cA, cB = _info['opened_wall']
    print(f"\n  Opened wall  : {cA} <-> {cB}  "
          f"(gain {_info['gain_cells']} cells, "
          f"saves {_info['improvement']} step(s))")
    print(f"  Path 1 steps : {_info['path1_steps']}  "
          f"({_info['path1_length']:.1f} units)")
    print(f"  Path 2 steps : {_info['path2_steps']}  "
          f"({_info['path2_length']:.1f} units)")
    print(f"  Ratio        : {_info['ratio']:.3f}x")
