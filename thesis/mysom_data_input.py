import numpy as np
from mysom.mysom import MySom
from pathlib import Path

x_path = Path('')
y_path = Path('')
figpath = Path('img_blinded_cell_line_data/')
datestr = 'YYYY_MM_DD'

x_data = np.genfromtxt(x_path, delimiter=',')
y_data = np.genfromtxt(y_path, delimiter=',')

label_list = ['Blinded Data']  # default is 'Blinded Data'
marker_list = ['o']
colour_list = ['#FFA500']

'''
Default values
--------------
x by y: 5*SQRT(n)
sigma:  1.0
learning rate:  0.5
random seed:    1
'''

som0 = MySom(x=, y=, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=1)

# initiate and plot each SOM
som_list = [som0]
for som in som_list:
    som.frobenius_norm_normalisation(y_data)
    som.make_som(10000)
    som.make_labels(y_path, label_list, marker_list, colour_list)
    som.plot_som_umatrix(figpath, datestr)
    som.plot_som_scatter(figpath, datestr)
    som.plot_density_function(figpath, datestr)
    som.plot_neuron_activation_frequency(figpath, datestr)
    som.plot_errors(1000, figpath, datestr)
