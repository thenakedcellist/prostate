import numpy as np
from mysom.mysom import MySom
from pathlib import Path

x_path = Path('../../../data/practice_datasets/colours_dataset_large/colours.txt')
y_path = Path('../../../data/practice_datasets/colours_dataset_large/colours.txt')
figpath = Path('img/')
datestr = '2020_12_03'

x_data = np.genfromtxt(x_path, delimiter=',')
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 4))

label_list = ['R', 'G', 'B', 'C', 'M', 'Y', 'K']  # default is 'Blinded Data'
marker_list = ['o', 'o', 'o', '_', '_', '_', 'x']
colour_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# 5*SQRT(865) = 147.05 so 12 . 12 grid

'''
Default values
--------------
x by y: 5*SQRT(n)
sigma:  1.0
learning rate:  0.5
random seed:    1
'''

som0 = MySom(x=12, y=12, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=1)

# initiate and plot each SOM
som_list = [som0]
for som in som_list:
    som.som_setup(x_data, y_data)
    som.frobenius_norm_normalisation(y_data)
    som.train_som(10000)
    som.plot_som_umatrix(figpath, datestr)
    som.plot_som_scatter(figpath, datestr)
    som.plot_density_function(figpath, datestr)
    som.plot_node_activation_frequency(figpath, datestr)
    som.plot_errors(1000, figpath, datestr)
