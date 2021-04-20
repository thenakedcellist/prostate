import numpy as np
from mysom.mysom import MySom
from pathlib import Path

x_path = Path('../../../data/yvette_20_09_02/xwavehw.csv')
y_path = Path('../../../data/yvette_20_11_18/shuffled_data_named.csv')
figpath = Path('img_unblinded_cell_line_data/')
datestr = '2020_12_07'

x_data = np.genfromtxt(x_path, delimiter=',')
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 1057))

label_list = ['(112)', 'PNT2', 'LNCaP']
marker_list = ['d', 'o', 'x']
colour_list = ['#00BFFF', '#FFA500', 'g']

# 5 * sqrt(285) = 84.41 so x = 9 y = 9 input_len = y_data.shape[1]
# or x=11 y=8

'''
Default values
--------------
x by y: 9 x 9
    test:   11 x 8
sigma:  1.0
    test:   0.5, 2.0, 4.0
learning rate:  0.5
    test: 0.1, 0.3, 0.75
random seed:    1
'''

som8 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=1)

som_list = [som8]
for som in som_list:
    som.frobenius_norm_normalisation(y_data)
    som.make_som(10000)
    som.make_labels(y_path, label_list, marker_list, colour_list)
    som.plot_som_umatrix(figpath, datestr, onlyshow=True)
    som.plot_som_scatter(figpath, datestr, onlyshow=True)
    som.plot_density_function(figpath, datestr, onlyshow=True)
    som.plot_node_activation_frequency(figpath, datestr, onlyshow=True)
    som.plot_errors(1000, figpath, datestr, onlyshow=True)
