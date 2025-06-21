from faabp_sim import extract_simulation_data, create_payload_animation, default_payload_params
import os

params = default_payload_params(curvity_on=0.4, curvity_off=-0.4, payload_radius=18.7)
# # data = extract_simulation_data('D:/ThesisData/data/dynamic_curvity/sim_data_cOn_1_cOff_0_pradius_18.7.npz')
# data = extract_simulation_data('D:/ThesisData/data/dynamic_v0/sim_data_vOn_10_vOff_3.75_curvity_0.6.npz')
# print(data)
# create_payload_animation(data['positions'], data['orientations'], data['velocities'], data['payload_positions'], params, 
#                          data['curvity_values'], f'D:/ThesisData/visualizations/dynamic_v0/sim_animation_vOn_10_vOff_3.75_curvity_0.6.mp4')

def query_data():
    count = 0
    root_dir = 'D:/ThesisData/data/dynamic_curvity'
    for file in os.listdir(root_dir):
        # get the file name without the extension
        file_name = os.path.splitext(file)[0]
        # get the parameters from the file name
        params = file_name.split('_')
        # index 3 = curvity_on, index 5 = curvity_off, index 7 = payload_radius
        curvity_on = float(params[3])
        curvity_off = float(params[5])
        payload_radius = float(params[7])
        # we want to find all files where curvity_on > curvity_off; AND both values are non-negative
        if curvity_on > curvity_off and curvity_on >= 0 and curvity_off >= 0:
            count += 1
            # data = extract_simulation_data(os.path.join(root_dir, file))
            # create_payload_animation(data['positions'], data['orientations'], data['velocities'], data['payload_positions'], params, 
            #              data['curvity_values'], f'D:/ThesisData/visualizations/dynamic_curvity/sim_animation_cOn_{curvity_on}_cOff_{curvity_off}_pradius_{payload_radius}.mp4')
    print(count)

target = 'D:/ThesisData/data/dynamic_curvity/sim_data_cOn_-0.2_cOff_-0.4_pradius_18.7.npz'
result = 'D:/ThesisData/visualizations/dynamic_curvity/sim_animation_cOn_-0.2_cOff_-0.4_pradius_18.7.mp4'

data = extract_simulation_data(target)
create_payload_animation(data['positions'], data['orientations'], data['velocities'], data['payload_positions'], params, 
                        data['curvity_values'], result)
