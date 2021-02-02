import numpy as np
from mysom.mysom import MySom
from pathlib import Path

x_path = Path('../../../../data/yvette_20_09_02/xwavehw.csv')
y_path = Path('../../../../data/yvette_20_09_02/High Wavenumbers for Dan.csv')
figpath = Path('img')
datestr = '2020_12_02'

x_data = np.genfromtxt(x_path, delimiter=',')
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 1057))

label_list = ['PNT2', 'LNCaP']  # default is 'Blinded Data'
marker_list = ['o', 'o']
colour_list = ['#FFA500', 'g']

# 5 * sqrt(30) = 27.39 so x = 6 y = 5 input_len = y_data.shape[1]

# default som
som = MySom(x=6, y=5, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=1)

som.som_setup(x_data, y_data)
som.frobenius_norm_normalisation(y_data)
som.train_som(10000)
som.plot_som_umatrix(figpath, datestr)
som.plot_som_scatter(figpath, datestr)
som.plot_density_function(figpath, datestr)
som.plot_neuron_activation_frequency(figpath, datestr)
som.plot_errors(1000, figpath, datestr)
