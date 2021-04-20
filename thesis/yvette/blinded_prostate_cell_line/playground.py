import numpy as np
from mysom.mysom import MySom
from pathlib import Path
import datetime as dt

p = dt.datetime.now()
print("Start time is " + str(p))

x_path = Path('../../../../data/yvette_20_09_02/xwavehw.csv')
y_path = Path('../../../../data/yvette_20_11_18/shuffled_data_named.csv')
figpath = Path('img_som_param_optimisation/')
datestr = 'som_param_optimisation_unblind'

x_data = np.genfromtxt(x_path, delimiter=',')
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 1057))

label_list = ['PNT2', 'LNCaP']
marker_list = ['o', 'x']
colour_list = ['#FFA500', '#FF00FF']

# 5 * sqrt(285) = 84.41 so x = 9 y = 9 input_len = y_data.shape[1]
# or x=11 y=8

# noisy signal is index 46
removal_list = [46]

'''
Test values
--------------
x by y:         17 x 5, 18 x 5, 19 x 5, 12 x 7, 22 x 6
sigma:          0.1, 0.5, 1.0, 2.0, 4.0
learning rate:  0.01, 0.1, 0.5, 0.9, 0.99
random seed:    1-10
'''

for i in range(1, 2):
    som0 = MySom(x=23, y=6, input_len=y_data.shape[1], sigma=3.0, learning_rate=0.75, topology='rectangular', random_seed=i)
    som0.som_setup(x_data, y_data, y_path, label_list, marker_list, colour_list)
    som0.remove_observations_from_input_data(removal_list)
    som0.frobenius_norm_normalisation()
    som0.train_som(100000)
    som0.plot_som_scatter(figpath, datestr)
    som0.plot_density_function(figpath, datestr)
    print("quantisation error " + str(som0.quantisation_err))
    print("topographic error " + str(som0.topographic_err))

#winmap = som0.som.win_map(som0.nydata, return_indices=True)
#poi = winmap[(0, 0)]Som
from pathlib import Path
