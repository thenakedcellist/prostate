import numpy as np
from mysom.mysom import MySom
from pathlib import Path

x_path = Path('../../../data/yvette_20_09_02/xwavehw.csv')
y_path = Path('../../../data/yvette_20_09_02/High Wavenumbers for Dan.csv')
figpath = Path('img/mysom_playground/')
datestr = '2018-12-02'

x_data = np.genfromtxt(x_path, delimiter=',')
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 1057))

label_list = ['PNT2', 'LNCaP']
marker_list = ['o', 'o']
colour_list = ['#FFA500', '#FFA500']

# 5 * sqrt(2640) = 256.90 so x = 16 y = 16 input_len = ydata.shape[1]

som = MySom(x=6, y=5, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=1)
som.frobenius_norm_normalisation(y_data)
som.make_som(10000)
som.make_labels(y_path, label_list, marker_list, colour_list)
som.plot_som_umatrix(figpath, datestr)
som.plot_som_scatter(figpath, datestr)
som.plot_density_function(figpath, datestr)
som.plot_neuron_activation_frequency(figpath, datestr)
som.plot_errors(1000, figpath, datestr)
