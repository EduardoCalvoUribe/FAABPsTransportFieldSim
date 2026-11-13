"""Utilities for the wall-pruning experiment (fig6).

Provides functions to enumerate interior wall segments in a Wilson maze and
progressively open (prune) them, so each run sees a different wall density.

Interior walls are identified by keys:
    ('h', r, c)  — horizontal wall between rows r and r+1 in column c
    ('v', r, c)  — vertical wall between columns c and c+1 in row r

When a wall key is pruned, the corresponding passage is opened in the passages
dict. Calling maze_to_walls() with the updated passages then automatically
removes the straight wall segment AND all arc segments at the affected corners,
because arc generation re-checks which walls are present at each junction.
"""

import copy
import random


def get_interior_wall_keys(passages, grid_size):
    """Return list of all interior wall keys present in `passages`.

    A perfect G×G maze always has (G-1)² interior walls.
    """
    walls = []
    for r in range(grid_size - 1):
        for c in range(grid_size):
            if (r + 1, c) not in passages.get((r, c), set()):
                walls.append(('h', r, c))
    for r in range(grid_size):
        for c in range(grid_size - 1):
            if (r, c + 1) not in passages.get((r, c), set()):
                walls.append(('v', r, c))
    return walls


def _open_passage(passages, wall_key):
    """Open the passage corresponding to wall_key in-place."""
    kind, r, c = wall_key
    if kind == 'h':
        passages.setdefault((r, c), set()).add((r + 1, c))
        passages.setdefault((r + 1, c), set()).add((r, c))
    else:  # 'v'
        passages.setdefault((r, c), set()).add((r, c + 1))
        passages.setdefault((r, c + 1), set()).add((r, c))


def build_pruned_passages(original_passages, wall_keys, n_prune, prune_seed):
    """Return a passages dict with n_prune walls removed.

    The wall removal order is a deterministic shuffle of wall_keys controlled by
    prune_seed, so every array task can independently reconstruct which walls are
    open at step n_prune without communicating with other tasks.

    Args:
        original_passages: passages dict from wilson.generate()
        wall_keys: list from get_interior_wall_keys() on the same passages
        n_prune: 0 → full maze; len(wall_keys) → all interior walls removed
        prune_seed: integer seed governing the shuffle

    Returns:
        New passages dict (deep copy) with the first n_prune walls opened.
    """
    rng = random.Random(prune_seed)
    shuffled = list(wall_keys)
    rng.shuffle(shuffled)

    passages = copy.deepcopy(original_passages)
    for key in shuffled[:n_prune]:
        _open_passage(passages, key)
    return passages
