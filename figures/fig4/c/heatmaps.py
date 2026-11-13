"""Fig 4c: payload-speed heatmaps. Writes into this folder."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))  # repo root, so `figures.fig4.code` imports work when run directly
import numpy as np
import matplotlib.pyplot as plt
from figures.fig4.code.paths import DATA_DIR, FIG_DIR, out_path


def dynamic_v0_payloadspeed():
    data_dir = f'{DATA_DIR}/dynamic_v0'
    data_dir_2 = f'{DATA_DIR}/dynamic_v0_2'
    
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
        plt.savefig(out_path(f'{FIG_DIR}/c/dynamicvO_heatmap_curvity_{c}.png'))
        plt.show()
    

def dynamic_curvity_payloadspeed():
    data_dir = f'{DATA_DIR}/dynamic_curvity'
    
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
        # for i_cOff, cOff_val in enumerate(unique_cOff):
        #     for i_cOn, cOn_val in enumerate(unique_cOn):
        #         if float(cOff_val) * float(c) < -1:
        #             plt.plot(i_cOn, i_cOff, 'wo', markersize=8, markerfacecolor='none')  # White plus sign
        #         if float(cOn_val) * float(c) < -1:
        #             plt.plot(i_cOn, i_cOff, 'wx', markersize=8)  # White x mark

        
        # plot
        plt.colorbar(label='⟨vp⟩/v0')
        plt.title(f'Average Payload Speed (α = {c})')
        plt.xlabel('Κ while having line-of-sight with goal') #'Curvity with Line-Of-Sight to Goal (x marker for ka < -1)'
        plt.ylabel('Κ while occluded by payload') #'Curvity while Occluded (o marker for ka < -1)'
        # plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels=unique_cOn)
        # plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels=unique_cOff)
        plt.tight_layout()
        plt.savefig(out_path(f'{FIG_DIR}/c/dynamiccurviy_heatmap_pradius_{c}.png'))
        plt.show()


if __name__ == "__main__":
    dynamic_curvity_payloadspeed()
    # dynamic_v0_payloadspeed()
