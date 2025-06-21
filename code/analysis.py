import numpy as np
import os
import matplotlib.pyplot as plt
from faabp_sim import create_payload_animation

def dynamic_v0_payloadspeed():
    data_dir = 'D:/ThesisData/data/dynamic_v0'
    data_dir_2 = 'D:/ThesisData/data/dynamic_v0_2'
    
    files = os.listdir(data_dir) + os.listdir(data_dir_2)
    
    avg_payload_speeds = np.zeros(shape=(len(files), 4))
    i = 0
    
    for file in files:
        if file.endswith('.npz'):
            if os.path.exists(os.path.join(data_dir, file)):
                current_dir = data_dir
            else:
                current_dir = data_dir_2
                
            # example file name: 'sim_data_vOn_3.75_vOff_3.75_curvity_0.2.npz'
            data = np.load(os.path.join(current_dir, file))
            vOn = file.split('_vOn_')[1].split('_vOff_')[0]
            vOff = file.split('_vOff_')[1].split('_curvity_')[0]
            curvity = file.split('_curvity_')[1].split('.npz')[0]
                        
            
            # average payload velocity for that run
            payload_velocities = data['payload_velocities']
            payload_speeds = np.linalg.norm(payload_velocities, axis=1)
            avg_payload_speeds[i][0] = curvity
            avg_payload_speeds[i][1] = vOn
            avg_payload_speeds[i][2] = vOff
            avg_payload_speeds[i][3] = np.mean(payload_speeds)
            i += 1
    

    
    results = np.array(avg_payload_speeds, dtype=float)

    unique_curvities = np.unique(results[:, 0])
    unique_vOn = np.unique(results[:, 1])
    unique_vOff = np.unique(results[:, 2])
    
    global_max_speed = results[:, 3].max()    

    for c in unique_curvities:
        mask = results[:, 0] == c
        subset = results[mask]

        heatmap = np.zeros((len(unique_vOff), len(unique_vOn)))
        
        for _, vOn_val, vOff_val, avg_speed in subset:
            i_vOn = np.where(unique_vOn == vOn_val)[0][0]
            i_vOff = np.where(unique_vOff == vOff_val)[0][0]
            heatmap[i_vOff, i_vOn] = avg_speed
        
        plt.figure(figsize=(8, 6))
        plt.imshow(
            heatmap,
            origin='lower',
            aspect='auto',
            cmap='viridis',
            # extent=[
            #     unique_vOn.min(), unique_vOn.max(),
            #     unique_vOff.min(), unique_vOff.max()
            # ],
            vmin=0,
            vmax=global_max_speed
        )
        # print(heatmap)
        
        # plot
        plt.colorbar(label='⟨vp⟩')
        plt.title(f'Average Payload Speed (Curvity = {c})')
        plt.xlabel('v0 during Line-Of-Sight with Goal')
        plt.ylabel('v0 during Occlusion')
        plt.xticks(ticks=[0, 1, 2], labels=unique_vOn)
        plt.yticks(ticks=[0, 1, 2], labels=unique_vOff)
        plt.tight_layout()
        plt.savefig(f'C:/Users/educa/Pictures/thesis/dynamic_v0/payload_speed_3/dynamicvO_heatmap_curvity_{c}.png')
        plt.show()
    

def dynamic_curvity_payloadspeed():
    data_dir = 'D:/ThesisData/data/dynamic_curvity'
    
    files = os.listdir(data_dir)
    
    avg_payload_speeds = np.zeros(shape=(len(files), 4))
    i = 0
    
    for file in files:
        if file.endswith('.npz'):
            # example file name: 'sim_data_cOn_1_cOff_-0.6_pradius_18.7.npz'
            data = np.load(os.path.join(data_dir, file))
            cOn = file.split('_cOn_')[1].split('_cOff_')[0]
            cOff = file.split('_cOff_')[1].split('_pradius_')[0]
            pradius = file.split('_pradius_')[1].split('.npz')[0]
                        
            # average payload velocity for that run
            payload_velocities = data['payload_velocities']
            payload_speeds = np.linalg.norm(payload_velocities, axis=1)
            avg_payload_speeds[i][0] = pradius
            avg_payload_speeds[i][1] = cOn
            avg_payload_speeds[i][2] = cOff
            avg_payload_speeds[i][3] = np.mean(payload_speeds)
            i += 1
    

    
    results = np.array(avg_payload_speeds, dtype=float)

    unique_pradius = np.unique(results[:, 0])
    unique_cOn = np.unique(results[:, 1])
    unique_cOff = np.unique(results[:, 2])
    
    global_max_speed = results[:, 3].max() / 3.75 # normalize by v0

    for c in unique_pradius:
        mask = results[:, 0] == c
        subset = results[mask]

        # Prepare a 2D array of shape (len(unique_vOff), len(unique_vOn))
        heatmap = np.zeros((len(unique_cOff), len(unique_cOn)))
        
        # Fill in each (vOff, vOn) cell with the corresponding avg_speed
        for _, cOn_val, cOff_val, avg_speed in subset:
            i_cOn = np.where(unique_cOn == cOn_val)[0][0]
            i_cOff = np.where(unique_cOff == cOff_val)[0][0]
            heatmap[i_cOff, i_cOn] = avg_speed / 3.75 # normalize by v0
        
        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(
            heatmap,
            origin='lower',
            aspect='auto',
            cmap='viridis',
            # extent=[
            #     unique_vOn.min(), unique_vOn.max(),
            #     unique_vOff.min(), unique_vOff.max()
            # ],
            vmin=0,
            vmax=global_max_speed
        )
        
        # tickmarks for cOff * pradius < -1
        for i_cOff, cOff_val in enumerate(unique_cOff):
            for i_cOn, cOn_val in enumerate(unique_cOn):
                if float(cOff_val) * float(c) < -1:
                    plt.plot(i_cOn, i_cOff, 'wo', markersize=8, markerfacecolor='none')  # White plus sign
                if float(cOn_val) * float(c) < -1:
                    plt.plot(i_cOn, i_cOff, 'wx', markersize=8)  # White x mark

        
        # plot
        plt.colorbar(label='⟨vp⟩/v0')
        plt.title(f'Average Payload Speed (Payload Radius = {c})')
        plt.xlabel('Curvity with Line-Of-Sight to Goal (x marker for ka < -1)')
        plt.ylabel('Curvity while Occluded (o marker for ka < -1)')
        plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels=unique_cOn)
        plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels=unique_cOff)
        plt.tight_layout()
        plt.savefig(f'C:/Users/educa/Pictures/thesis/dynamiccurviy_heatmap_pradius_{c}.png')
        plt.show()


    

def unravel_trajectory(trajectory, box_size=350):
    """Unravel a trajectory that was created with periodic boundary conditions.
    
    Args:
        trajectory: numpy array of shape (n_steps, 2) containing x,y coordinates
        box_size: float, the size of the periodic box
        
    Returns:
        numpy array of shape (n_steps, 2) containing the unraveled trajectory
    """
    # Create a copy to avoid modifying the original
    unraveled = trajectory.copy()
    
    # Track jump locations for both x and y coordinates
    jump_pos_x = []
    jump_pos_y = []
    
    # Go through trajectory in pairs
    for i in range(len(trajectory) - 1):
        # Check x coordinate
        if abs(trajectory[i+1, 0] - trajectory[i, 0]) > box_size/2:
            if trajectory[i, 0] > trajectory[i+1, 0]:
                jump_pos_x.append((i, 1))  # Jumped at upper boundary
            else:
                jump_pos_x.append((i, -1))  # Jumped at lower boundary
                
        # Check y coordinate
        if abs(trajectory[i+1, 1] - trajectory[i, 1]) > box_size/2:
            if trajectory[i, 1] > trajectory[i+1, 1]:
                jump_pos_y.append((i, 1))  # Jumped at upper boundary
            else:
                jump_pos_y.append((i, -1))  # Jumped at lower boundary
    
    # Update values for x coordinate
    for i, direction in jump_pos_x:
        unraveled[i+1:, 0] += direction * box_size
        
    # Update values for y coordinate
    for i, direction in jump_pos_y:
        unraveled[i+1:, 1] += direction * box_size
        
    return unraveled

def graph_pathway():
    data_dir = 'D:/ThesisData/data/dynamic_v0'
    data_dir_2 = 'D:/ThesisData/data/dynamic_v0_2'
    
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
    goal_circle = plt.Circle(goal_position, 10, color='darkgreen', alpha=0.6, label='Goal', 
                           edgecolor='darkgreen', facecolor='darkgreen')
    ax.add_patch(goal_circle)

    ax.set_xlim(-400, 350)
    ax.set_ylim(-500, 350)
    ax.set_title('Payload Paths for Dynamic v0')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    
    ax.set_aspect('equal')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Curvity')
    
    plt.tight_layout()
    plt.savefig('C:/Users/educa/Pictures/thesis/paths/payload_pathway_full.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # graph_pathway()
    dynamic_v0_payloadspeed()
    # data = np.load(f'D:/ThesisData/data/dynamic_v0/sim_data_vOn_{3.75}_vOff_{10}_curvity_{0.6}.npz')
    # n_saved_states = len(data['positions'])
    # original_steps = (n_saved_states - 1) * 10
    # print(f"Original simulation ran for {original_steps} steps")