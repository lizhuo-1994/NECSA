import matplotlib.pyplot as plt
import numpy as np
import os,sys
import json

curves_fir = '../curves'

def load_data(policy, env):
    data_file = f"{curves_fir}/{env}"
    file_list = os.listdir(data_file)
    data_list = []
    for json_file in file_list:
        with open(data_file + '/' + json_file, 'r') as f:
            data = json.load(f)
        data = np.array(data)[:,2]
        data_list.append(data)

    return data_list
data_list = load_data('', 'Hopper-v3')     
data_list = np.array(data_list)


mean = np.mean(data_list, axis = 0)
std = np.std(data_list, axis = 0)
print(std)
'''
x = np.arange(len(mean_1))
plt.plot(x, mean_1, 'b-', label='mean_1')
plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
plt.legend()
plt.show()
'''