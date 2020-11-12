import numpy as np
from mysom.mysom import MySom
from pathlib import Path

x_path = Path('../../data/banbury/_data_1-6/_wavenumbers.csv')
y_path = Path('../../data/banbury/_data_1-6/all_data.csv')
figpath = Path('banbury_playground/')
datestr = '2018-11-09'

x_data = np.genfromtxt(x_path, delimiter=',', usecols=np.arange(1, 1016))
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 1016))

label_list = ['cornea', 'lens', 'optic_nerve', 'retina', 'vitreous']
marker_list = ['o', 'o', 'o', 'o', 'o']
colour_list = ['r', 'g', 'b', 'c', 'm']

# 5 * sqrt(2640) = 256.90 so x = 16 y = 16 input_len = ydata.shape[1]

som = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, random_seed=1)
som.scikit_norm(y_data)
som.make_som(10000)
som.make_labels(y_path, label_list, marker_list, colour_list)
som.plot_som_umatrix(figpath, datestr)
som.plot_som_scatter(figpath, datestr)
som.plot_neuron_activation_frequency(figpath, datestr)
som.plot_errors(1000, figpath, datestr)
som.plot_density_function(figpath, datestr)
