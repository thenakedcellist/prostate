import numpy as np
from mysom.mysom import MySom
from pathlib import Path
import csv
import datetime as dt

p = dt.datetime.now()
print("Start time is " + str(p))

x_path = Path('../../../../data/yvette_20_09_02/xwavehw.csv')
y_path = Path('../../../../data/yvette_20_11_18/shuffled_data_unnamed.csv')
figpath = Path('img_blinded_soms_outlier_removed/')
datestr = '2021_03_18'

x_data = np.genfromtxt(x_path, delimiter=',')
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(0, 1056))

label_list = ['Blinded Data']
marker_list = ['o']
colour_list = ['#FFA500']

# 5 * sqrt(285) = 84.41

# noisy signal is index 46
removal_list = [46]

'''
Test values
--------------
x by y:         19 x 5, 14 x 6, 17 x 5, 15 x 4, 23 x 6
sigma:          0.5, 1.0, 2.0, 3.0, 4.0
learning rate:  0.01, 0.1, 0.5, 0.9, 0.99
random seed:    1-10
'''

# som families organised as lists of family members per family, and a list of families
som_family_numbers = list(np.arange(1, 126))
som_family_members = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
som_family_list = [[f'{i}{j}' for j in som_family_members] for i in som_family_numbers]
som_pop_list = [f'{i}{j}' for i in som_family_numbers for j in som_family_members]

# parameter lists
map_dimensions_list = [(19, 5), (14, 6), (17, 5), (15, 4), (23, 6)]
sigma_list = [0.5, 1.0, 2.0, 3.0, 4.0]
learning_rate_list = [0.5, 0.75, 0.9, 0.95, 0.99]
random_seed_list = list(range(1, 11))
all_params_list = []
for a in map_dimensions_list:
    for b in sigma_list:
        for c in learning_rate_list:
            for d in random_seed_list:
                all_params_list.append((a, b, c, d))

# dict with som family key and value parameter tuple
family_dict = {som_pop_list[i]: all_params_list[i] for i in range(len(som_pop_list))}

# list with som family, parameters, and som
all_soms_list = list()
for k, v in family_dict.items():
    all_soms_list.append(MySom(x=v[0][0], y=v[0][1], input_len=y_data.shape[1], sigma=v[1],
                               learning_rate=v[2], topology='rectangular', random_seed=v[3]))
full_som_list = [[som_pop_list[i], all_params_list[i], all_soms_list[i]] for i in range(len(som_pop_list))]

# batch train soms and append q_err adn t_err to full_som_list index
counter = 0
for v in full_som_list:
    v[2].som_setup(x_data, y_data, y_path, label_list, marker_list, colour_list)
    v[2].remove_observations_from_input_data(removal_list)
    v[2].frobenius_norm_normalisation()
    v[2].train_som(100000)
    v[2].plot_som_umatrix(figpath, datestr)
    v[2].plot_som_scatter(figpath, datestr)
    v[2].plot_node_activation_frequency(figpath, datestr)
    v[2].plot_density_function(figpath, datestr)
    v.append(v[2].quantisation_err)
    v.append(v[2].topographic_err)
    counter += 1
    print(f'Iteration {counter} complete')

# add headings to first row of full_som_list
headings = ['family_member', 'parameters', 'som', 'q_err', 't_err']
full_som_list.insert(0, headings)

# save full_som_list to outfile csv
with open("blinded_soms_outlier_removed_parameters.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(full_som_list)

q = dt.datetime.now()
print("Stop time is " + str(q))

process_time = q - p
print("Runtime is " + str(process_time))
