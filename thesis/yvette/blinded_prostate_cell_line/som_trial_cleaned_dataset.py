import numpy as np
from mysom.mysom import MySom
from pathlib import Path


x_path = Path('../../../../data/yvette_20_09_02/xwavehw.csv')
y_path = Path('../../../../data/yvette_20_11_18/shuffled_data_unnamed.csv')
figpath = Path('img_som_trial/')
datestr = 'som_trial_augmented_dataset'

x_data = np.genfromtxt(x_path, delimiter=',')
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(0, 1056))

label_list = ['Blinded Data']
marker_list = ['o']
colour_list = ['#FFA500']

# 5 * sqrt(285) = 84.41
# removal list
removal_list = [46]

for i in range(1, 2):
    som0 = MySom(x=19, y=5, input_len=y_data.shape[1], sigma=2.0, learning_rate=0.9, topology='rectangular', random_seed=i)
    som0.som_setup(x_data, y_data, y_path, label_list, marker_list, colour_list)
    som0.remove_observations_from_input_data(removal_list)
    som0.frobenius_norm_normalisation()
    som0.train_som(100000)
    som0.plot_som_scatter(figpath, datestr)
    som0.plot_density_function(figpath, datestr)
    print("quantisation error " + str(som0.quantisation_err))
    print("topographic error " + str(som0.topographic_err))

winmap = som0.som.win_map(som0.nydata, return_indices=True)
