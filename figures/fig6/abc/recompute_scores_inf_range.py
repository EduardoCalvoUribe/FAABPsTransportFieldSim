"""Recompute particle scores with a larger observation range (VIEW_RANGE).

For each maze_G*_run1.npz in this folder, writes maze_G*_run1_infrange.npz (name
kept from the earlier r = infinity version): a copy with `particle_scores`
replaced by the converged score under r = VIEW_RANGE (currently 180).

Slightly modified score rule (original is src/simulation.py::point_polarity_to_goal):
  * score 0 if the goal is within VIEW_RANGE (minimum-image distance) and the
    particle has line of sight to it (no wall in the way; the payload does NOT
    block; direct line, no periodic wrapping)
  * otherwise 1 + min score among particles within VIEW_RANGE it can see
    (minimum-image segment not crossed by any wall; payload does not block)
  * 9999 if no chain of visible particles reaches a score-0 particle
Since the rule is a one-hop-per-update relaxation, its fixed point is the BFS
hop distance from the score-0 particles over the visibility graph.
"""

import glob
import os
import sys
import time

import numpy as np
from numba import njit

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _ROOT)

from src.forces import build_wall_spatial_index
from src.physics_utils import line_intersects_any_wall_indexed, compute_minimum_distance

UNREACHABLE = 9999
VIEW_RANGE = 180.0   # observation range r used for the recomputation (original sim: 90)


@njit(cache=True)
def bfs_scores(positions, goal, payload_pos, payload_radius, walls, box_size,
               grid_offsets, grid_indices, n_wall_cells, wall_cell_size, view_range):
    n = positions.shape[0]
    r2 = view_range * view_range
    scores = np.full(n, UNREACHABLE, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0

    for i in range(n):
        # Goal must be within view_range (minimum-image distance).
        r_g = compute_minimum_distance(positions[i], goal, box_size)
        if r_g[0] * r_g[0] + r_g[1] * r_g[1] > r2:
            continue
        # Payload does NOT block line of sight to the goal; only walls do.
        if not line_intersects_any_wall_indexed(
                positions[i, 0], positions[i, 1], goal[0], goal[1], walls,
                grid_offsets, grid_indices, n_wall_cells, wall_cell_size):
            scores[i] = 0
            queue[tail] = i
            tail += 1

    while head < tail:
        u = queue[head]
        head += 1
        for v in range(n):
            if scores[v] != UNREACHABLE:
                continue
            r_uv = compute_minimum_distance(positions[u], positions[v], box_size)
            if r_uv[0] * r_uv[0] + r_uv[1] * r_uv[1] > r2:
                continue
            blocked = line_intersects_any_wall_indexed(
                positions[u, 0], positions[u, 1],
                positions[u, 0] + r_uv[0], positions[u, 1] + r_uv[1],
                walls, grid_offsets, grid_indices, n_wall_cells, wall_cell_size)
            if not blocked:
                scores[v] = scores[u] + 1
                queue[tail] = v
                tail += 1
    return scores


def process(path):
    data = dict(np.load(path, allow_pickle=True))
    positions = np.ascontiguousarray(data["positions"][-1], dtype=np.float64)
    walls = np.ascontiguousarray(data["walls"], dtype=np.float64)
    box_size = float(data["box_size"])
    payload_radius = float(data["payload_radius"])
    goal = np.ascontiguousarray(data["goal_position"], dtype=np.float64)
    payload_pos = np.ascontiguousarray(data["payload_positions"][-1], dtype=np.float64)

    wall_cell_size = max(box_size / 20.0, 2.0)
    offsets, indices, n_cells = build_wall_spatial_index(
        walls, box_size, wall_cell_size, payload_radius)

    new_scores = bfs_scores(positions, goal, payload_pos, payload_radius, walls, box_size,
                            offsets, indices, n_cells, wall_cell_size, VIEW_RANGE)

    old = data["particle_scores"][-1]
    print(f"  old: min={old.min()} median={int(np.median(old))} max={old.max()} "
          f"unreachable={(old >= UNREACHABLE).sum()}")
    print(f"  new: min={new_scores.min()} median={int(np.median(new_scores))} "
          f"max={new_scores.max()} unreachable={(new_scores >= UNREACHABLE).sum()}")

    data["particle_scores"] = new_scores[np.newaxis].astype(data["particle_scores"].dtype)
    out = path.replace(".npz", "_infrange.npz")
    np.savez(out, **data)
    return out


if __name__ == "__main__":
    files = sorted(f for f in glob.glob(os.path.join(_HERE, "maze_G*_run1.npz")))
    for f in files:
        t0 = time.time()
        print(f"{os.path.basename(f)}")
        out = process(f)
        print(f"  saved {os.path.basename(out)} ({time.time() - t0:.1f}s)")
