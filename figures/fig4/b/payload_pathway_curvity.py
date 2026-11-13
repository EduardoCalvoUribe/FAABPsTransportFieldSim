"""Fig 4b: payload paths for the dynamic-curvity sweep. Writes into this folder."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))  # repo root, so `figures.fig4.code` imports work when run directly
import numpy as np
import matplotlib.pyplot as plt
from figures.fig4.code.paths import DATA_DIR, FIG_DIR, out_path
from figures.fig4.code.trajectory_utils import unravel_trajectory, mark_goal_start_box


def graph_pathway_curvity(payload_radii=None):
    """Payload paths for the dynamic-curvity sweep (one figure per payload radius).

    Mirrors graph_pathway(), but for files named
    sim_data_cOn_<cOn>_cOff_<cOff>_pradius_<r>.npz. Each path is coloured by
    Δκ = cOn - cOff (curvity with line-of-sight minus curvity while occluded).
    """
    import re
    data_dir = f'{DATA_DIR}/dynamic_curvity'
    pattern = re.compile(r'sim_data_cOn_(.+)_cOff_(.+)_pradius_(.+)\.npz')

    entries = []  # (file, cOn, cOff, pradius)
    for file in os.listdir(data_dir):
        m = pattern.fullmatch(file)
        if m:
            entries.append((file, float(m.group(1)), float(m.group(2)), float(m.group(3))))

    radii = sorted({e[3] for e in entries})
    if payload_radii is not None:
        radii = [r for r in radii if r in payload_radii]

    box_size = 350
    goal_position = [4*(box_size / 5), 4*(box_size / 5)]
    norm = plt.Normalize(-2, 2)
    cmap = plt.cm.viridis

    for r in radii:
        group = sorted((e for e in entries if e[3] == r), key=lambda e: e[1] - e[2])
        fig, ax = plt.subplots(figsize=(10, 8))
        for file, c_on, c_off, _ in group:
            data = np.load(os.path.join(data_dir, file))
            payload_positions_unraveled = unravel_trajectory(data['payload_positions'])
            ax.plot(payload_positions_unraveled[:, 0], payload_positions_unraveled[:, 1],
                    color=cmap(norm(c_on - c_off)), alpha=0.4)

        mark_goal_start_box(ax, box_size, goal_position)
        ax.set_title(f'Payload Paths for Dynamic Curvity (Payload Radius = {r})')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 350)
        ax.set_ylim(0, 350)
        ax.set_aspect('equal')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Δκ', shrink=0.8, aspect=20)
        plt.tight_layout()
        plt.savefig(out_path(f'{FIG_DIR}/b/payload_pathway_curvity_pradius_{r}.png'), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    graph_pathway_curvity()
