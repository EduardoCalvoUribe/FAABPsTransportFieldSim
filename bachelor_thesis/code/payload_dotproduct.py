
import numpy as np
import argparse
import sys
import os
import matplotlib.pyplot as plt
import re
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from payload_analysis import unravel_trajectory

def parse_filename_parameters_velocity(filename):
    if filename.endswith('.npz'):
        filename = filename[:-4] 
        
    parts = filename.split('_')
    
    vOn = None
    vOff = None
    curvity = None
    
    for i, part in enumerate(parts):
        if part == 'vOn' and i + 1 < len(parts):
            try:
                vOn = float(parts[i + 1])
            except ValueError:
                raise ValueError(f"Invalid vOn value: {parts[i + 1]}")
        elif part == 'vOff' and i + 1 < len(parts):
            try:
                vOff = float(parts[i + 1])
            except ValueError:
                raise ValueError(f"Invalid vOff value: {parts[i + 1]}")
        elif part == 'curvity' and i + 1 < len(parts):
            try:
                curvity = float(parts[i + 1])
            except ValueError:
                raise ValueError(f"Invalid curvity value: {parts[i + 1]}")
    
    if vOn is None or vOff is None or curvity is None:
        raise ValueError(f"Could not find parameters in filename: {filename}")
    
    return vOn, vOff, curvity

def parse_filename_parameters_curvity(filename):
    if filename.endswith('.npz'):
        filename = filename[:-4]  
    
    parts = filename.split('_')
    
    cOn = None
    cOff = None
    pradius = None
    
    for i, part in enumerate(parts):
        if part == 'cOn' and i + 1 < len(parts):
            try:
                cOn = float(parts[i + 1])
            except ValueError:
                raise ValueError(f"Invalid cOn value: {parts[i + 1]}")
        elif part == 'cOff' and i + 1 < len(parts):
            try:
                cOff = float(parts[i + 1])
            except ValueError:
                raise ValueError(f"Invalid cOff value: {parts[i + 1]}")
        elif part == 'pradius' and i + 1 < len(parts):
            try:
                pradius = float(parts[i + 1])
            except ValueError:
                raise ValueError(f"Invalid pradius value: {parts[i + 1]}")
    
    if cOn is None or cOff is None or pradius is None:
        raise ValueError(f"Could not find parameters in filename: {filename}")
    
    return cOn, cOff, pradius


def load_simulation_data(filename):
    data = np.load(filename)
    return data


def calculate_path_dot_products(payload_positions, n_steps, box_size=350):

    unraveled_positions = unravel_trajectory(payload_positions, box_size)
        
    total_frames = len(unraveled_positions)
    frames_per_segment = total_frames // n_steps

    dot_products = []
    for i in range(n_steps):
        start_idx = i * frames_per_segment
        end_idx = (i + 1) * frames_per_segment if i < n_steps - 1 else total_frames
        
        start_pos = unraveled_positions[start_idx]
        end_pos = unraveled_positions[end_idx - 1]  # using end_idx - 1 to avoid overlap
        
        displacement_vector = end_pos - start_pos
        
        displacement_magnitude = np.linalg.norm(displacement_vector)
        
        if displacement_magnitude > 1e-12:
            normalized_displacement = displacement_vector / displacement_magnitude
        else:
            normalized_displacement = np.array([1.0, 0.0]) # some direction
        
        
        # goal vector is the "correct direction" 
        # vector from the start of the segment to the goal
        goal_pos = np.array([4*(box_size / 5), 4*(box_size / 5)])
        
        goal_vector = goal_pos - start_pos
        goal_vector_magnitude = np.linalg.norm(goal_vector)
        if displacement_magnitude > 1e-12:
            normalized_goal = goal_vector / goal_vector_magnitude
        else:
            normalized_goal = np.array([np.sqrt(2)/2, np.sqrt(2)/2]) #45
        
        
        dot_product = np.dot(normalized_displacement, normalized_goal)
        dot_products.append(dot_product)
        
    return dot_products


def run_dotproduct_analysis(filename, n_steps=20):

    box_size = 350

    if not os.path.exists(filename):
        print(f"Error: File {filename} does not exist")
        return
    
    data = load_simulation_data(filename)
    
    if data is None:
        print("Failed to load simulation data")
        return
    
    if 'payload_positions' not in data:
        print("Error: 'payload_positions' not found in the data file")
        print(f"Available keys: {list(data.keys())}")
        return
    
    payload_positions = data['payload_positions']
    
    dot_products = calculate_path_dot_products(payload_positions, n_steps, box_size)
    
    return dot_products
    
def main_curvity():
    pradius_separated_data = {
        18.7: {'mean_dot_products': [], 'std_dot_products': [], 'cOn': [], 'cOff': [], 'filesize': []},
        14.4: {'mean_dot_products': [], 'std_dot_products': [], 'cOn': [], 'cOff': [], 'filesize': []},
        10.1: {'mean_dot_products': [], 'std_dot_products': [], 'cOn': [], 'cOff': [], 'filesize': []},
        5.8: {'mean_dot_products': [], 'std_dot_products': [], 'cOn': [], 'cOff': [], 'filesize': []},
        1.5: {'mean_dot_products': [], 'std_dot_products': [], 'cOn': [], 'cOff': [], 'filesize': []}
    }
    
    file_size_dict = {}
    
    for file in os.listdir("D:/ThesisData/data/dynamic_curvity"):
        filename = f"D:/ThesisData/data/dynamic_curvity/{file}"
        # read size of file and save to a dictionary
        file_size = os.path.getsize(filename)
        cOn, cOff, pradius = parse_filename_parameters_curvity(file)
        file_size_dict[file] = file_size
        dot_products = run_dotproduct_analysis(filename, 20) 

        pradius_separated_data[pradius]['mean_dot_products'].append(np.mean(dot_products))
        pradius_separated_data[pradius]['std_dot_products'].append(np.std(dot_products))
        pradius_separated_data[pradius]['cOn'].append(cOn)
        pradius_separated_data[pradius]['cOff'].append(cOff)
        pradius_separated_data[pradius]['filesize'].append(file_size)
    
    for i, pradius in enumerate(sorted(pradius_separated_data.keys())):
        # plt.subplot(3, 2, i+1)
        mean_dps = pradius_separated_data[pradius]['mean_dot_products']
        std_dps = pradius_separated_data[pradius]['std_dot_products']
        cOn_vals = pradius_separated_data[pradius]['cOn']
        cOff_vals = pradius_separated_data[pradius]['cOff']
        filesize_vals = pradius_separated_data[pradius]['filesize']
        
        # Calculate cOn - cOff for coloring
        curvature_diff = [cOn - cOff for cOn, cOff in zip(cOn_vals, cOff_vals)]
        
        # Sort all arrays by curvature_diff
        sorted_data = sorted(zip(mean_dps, std_dps, cOn_vals, cOff_vals, filesize_vals, curvature_diff), 
                           key=lambda x: x[5])  
        
        # Unzip the sorted data
        mean_dps_sorted, std_dps_sorted, cOn_vals_sorted, cOff_vals_sorted, filesize_vals_sorted, curvature_diff_sorted = zip(*sorted_data)
        
        edge_colors = []
        for cOn, cOff, filesize in zip(cOn_vals_sorted, cOff_vals_sorted, filesize_vals_sorted):
            # print(f"cOn: {cOn}, cOff: {cOff}, filesize: {filesize}")
            if cOn > 0 and cOff > 0:
                edge_colors.append('red')
            elif filesize < 280243640: # 280243650
                edge_colors.append('green')
            else:
                edge_colors.append('none')
        
        scatter = plt.scatter(mean_dps_sorted, std_dps_sorted, c=curvature_diff_sorted, alpha=0.7, cmap='viridis', 
                            edgecolors=edge_colors, linewidth=1)
        plt.xlabel("Mean Cosine Similarity")
        plt.ylabel("Std Cosine Similarity")
        plt.ylim(-0.05, 1.05)
        plt.xlim(-1.05, 1.05)
            # # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('cOn - cOff')
        plt.gca().set_aspect(2, adjustable='box') # force square aspect ratio
        plt.title(f"Path-Goal Alignment, payload radius = {pradius}")
        plt.tight_layout(pad=2.0)
        plt.show()
    
    

def main_velocity():
    data = {
        'mean_dot_products': [],
        'std_dot_products': [],
        'vOn': [],
        'vOff': [],
        'curvity': [],
        'filesize': []
    }
    
    # Check both folders
    folders = ["D:/ThesisData/data/dynamic_v0", "D:/ThesisData/data/dynamic_v0_2"]
    
    for folder in folders:
        for file in os.listdir(folder):
            filename = f"{folder}/{file}"
            file_size = os.path.getsize(filename)
            vOn, vOff, curvity = parse_filename_parameters_velocity(file)
            dot_products = run_dotproduct_analysis(filename, 20) 
            data['mean_dot_products'].append(np.mean(dot_products))
            data['std_dot_products'].append(np.std(dot_products))
            data['vOn'].append(vOn)
            data['vOff'].append(vOff)
            data['curvity'].append(curvity)
            data['filesize'].append(file_size)
        

    vOn_vals = data['vOn']
    vOff_vals = data['vOff']
    curvities = [vOn - vOff for vOn, vOff in zip(vOn_vals, vOff_vals)]
    
    scatter = plt.scatter(data['mean_dot_products'], data['std_dot_products'], c=curvities, alpha=0.7, cmap='viridis', 
                         edgecolors='gray', linewidth=0.3)
    plt.xlabel("Mean Cosine Similarity")
    plt.ylabel("Std Cosine Similarity")
    plt.ylim(-0.05, 1.05)
    plt.xlim(-1.05, 1.05)
    cbar = plt.colorbar(scatter)
    cbar.set_label('vOn - vOff')
    plt.gca().set_aspect(2, adjustable='box') # force square aspect ratio
    plt.title("Path-Goal Alignment, Dynamic Velocity")
    plt.tight_layout(pad=2.0)
    plt.show()

if __name__ == "__main__":
    # main_curvity()
    main_velocity()
    
    
    
    # file_size_dict = dict(sorted(file_size_dict.items(), key=lambda item: item[1]))
    # count all files smaller than the largest file
    # count = 0
    # total_files = len(file_size_dict)
    # for file in file_size_dict:
    #     if file_size_dict[file] < file_size_dict[max(file_size_dict, key=file_size_dict.get)]:
    #         print(file_size_dict[file])
            # count += 1
    # print(f"Number of files smaller than the largest file: {count} out of {total_files}")
    # print(f"Largest file size: {file_size_dict[max(file_size_dict, key=file_size_dict.get)]}")