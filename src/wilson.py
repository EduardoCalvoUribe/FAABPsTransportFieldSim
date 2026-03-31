"""
Wilson's algorithm for uniform spanning tree maze generation.

Produces a perfect maze (spanning tree of a W×W grid) using loop-erased
random walks. Each generated maze is drawn uniformly from all spanning trees.

Data structure: adjacency sets
    passages: dict[tuple[int,int], set[tuple[int,int]]]
    passages[(r, c)] = set of grid neighbors connected by a carved passage.
"""

import random
from collections import defaultdict
from typing import Optional


def generate(width: int, height: Optional[int] = None, seed: Optional[int] = None) -> dict:
    """
    Run Wilson's algorithm on a width × height grid.

    Returns passages: dict mapping each cell (r, c) to the set of its
    connected neighbors.
    """
    if height is None:
        height = width
    rng = random.Random(seed)

    passages = defaultdict(set)

    def neighbors(r: int, c: int) -> list:
        result = []
        if r > 0:          result.append((r - 1, c))
        if r < height - 1: result.append((r + 1, c))
        if c > 0:          result.append((r, c - 1))
        if c < width - 1:  result.append((r, c + 1))
        return result

    all_cells = [(r, c) for r in range(height) for c in range(width)]

    # Seed the tree with one arbitrary cell.
    in_tree = {all_cells[0]}
    not_in_tree = list(all_cells[1:])
    rng.shuffle(not_in_tree)

    for start in not_in_tree:
        if start in in_tree:
            continue

        # Loop-erased random walk from `start` until hitting the tree.
        path = [start]
        visited_order = {start: 0}

        current = start
        while current not in in_tree:
            nxt = rng.choice(neighbors(*current))
            if nxt in visited_order:
                # Erase the loop: truncate path back to where nxt was last seen.
                loop_start = visited_order[nxt]
                for cell in path[loop_start + 1:]:
                    del visited_order[cell]
                path = path[:loop_start + 1]
            else:
                visited_order[nxt] = len(path)
                path.append(nxt)
            current = nxt

        # Carve passages along the loop-erased path.
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            passages[a].add(b)
            passages[b].add(a)
            in_tree.add(a)
        in_tree.add(path[-1])  # last cell connects to the existing tree

    return dict(passages)


def render_ascii(passages: dict, width: int, height: Optional[int] = None) -> str:
    """Return an ASCII art string of the maze."""
    if height is None:
        height = width

    lines = []
    # Top border
    lines.append("+" + ("+--" * width)[1:] + "-+")  # simplified below
    lines = ["+" + "--+" * width]

    for r in range(height):
        # Cell row: left border, then each cell + east wall
        row = "|"
        for c in range(width):
            east = (r, c + 1)
            row += "  " + (" " if east in passages.get((r, c), set()) else "|")
        lines.append(row)

        # South wall row
        wall = "+"
        for c in range(width):
            south = (r + 1, c)
            wall += ("  " if south in passages.get((r, c), set()) else "--") + "+"
        lines.append(wall)

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None

    p = generate(size, seed=seed)
    print(render_ascii(p, size))
