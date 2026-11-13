"""Fig 4d: payload paths for the dynamic-v0 sweep. Writes into this folder."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))  # repo root, so `figures.fig4.code` imports work when run directly
import numpy as np
import matplotlib.pyplot as plt
from figures.fig4.code.paths import DATA_DIR, FIG_DIR, out_path
from figures.fig4.code.trajectory_utils import unravel_trajectory, mark_goal_start_box


def graph_pathway():
    data_dir = f'{DATA_DIR}/dynamic_v0'
    data_dir_2 = f'{DATA_DIR}/dynamic_v0_2'
    
    files = os.listdir(data_dir) + os.listdir(data_dir_2)
    
    file_curvity_pairs = []
    for file in files:
        if file.endswith('.npz'):
            curvity = float(file.split('_curvity_')[1].split('.npz')[0])
            file_curvity_pairs.append((file, curvity))
    
    file_curvity_pairs.sort(key=lambda x: x[1])
    files = [pair[0] for pair in file_curvity_pairs]
    curvities = [pair[1] for pair in file_curvity_pairs]
    
    norm = plt.Normalize(min(curvities), max(curvities))
    cmap = plt.cm.viridis  
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for file in files:
        if file.endswith('.npz'):
            # Determine which directory the file is from
            if os.path.exists(os.path.join(data_dir, file)):
                current_dir = data_dir
            else:
                current_dir = data_dir_2
                
            # example file name: 'sim_data_vOn_3.75_vOff_3.75_curvity_0.2.npz'
            data = np.load(os.path.join(current_dir, file))
            # get positions
            payload_positions = data['payload_positions']
            payload_positions_unraveled = unravel_trajectory(payload_positions)
            
            # Extract curvity from filename and get color
            curvity = float(file.split('_curvity_')[1].split('.npz')[0])
            color = cmap(norm(curvity))
            
            ax.plot(payload_positions_unraveled[:, 0], payload_positions_unraveled[:, 1], 
                    color=color, alpha=0.4) 

    box_size = 350
    goal_position = [4*(box_size / 5), 4*(box_size / 5)]
    mark_goal_start_box(ax, box_size, goal_position)

    ax.set_xlim(0, 350)
    ax.set_ylim(0, 350)
    ax.set_title('Payload Paths for Dynamic v0')
    ax.grid(True, alpha=0.3)
    
    ax.set_aspect('equal')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='κ', shrink=0.8, aspect=20)
    
    plt.tight_layout()
    plt.savefig(out_path(f'{FIG_DIR}/d/payload_pathway_v0.png'), dpi=300)
    plt.show()


if __name__ == "__main__":
    graph_pathway()
