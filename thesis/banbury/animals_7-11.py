import numpy as np
from mysom.mysom import MySom
from pathlib import Path

x_path = Path('../../../data/banbury/_data_7-11/_wavenumbers.csv')
y_path = Path('../../../data/banbury/_data_7-11/all_data.csv')
figpath = Path('img_7-11/')
datestr = '2020_12_03'

x_data = np.genfromtxt(x_path, delimiter=',', usecols=np.arange(1, 1016))
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 1016))

label_list = ['cornea', 'lens', 'optic_nerve', 'retina', 'vitreous']  # default is 'Blinded Data'
marker_list = ['o', '_', 'x', '|', 'd']
colour_list = ['#20ABC4', '#2FAB40', '#101010', '#640098', '#E66510']

# 5*SQRT(2200) = 234.52 so 16 . 15 SOM

'''
Default values
--------------
x by y: 5*SQRT(n)
sigma:  1.0
learning rate:  0.5
random seed:    1
'''

som0 = MySom(x=16, y=15, input_len=y_data.shape[1], sigma=4.0, learning_rate=1.0, topology='rectangular', random_seed=1)

# initiate and plot each SOM
som_list = [som0]
for som in som_list:
    som.scikit_norm(y_data)
    som.make_som(10000)
    som.make_labels(y_path, label_list, marker_list, colour_list)
    som.plot_som_umatrix(figpath, datestr)
    som.plot_som_scatter(figpath, datestr)
    som.plot_density_function(figpath, datestr)
    som.plot_neuron_activation_frequency(figpath, datestr)
    #som.plot_errors(1000, figpath, datestr)
