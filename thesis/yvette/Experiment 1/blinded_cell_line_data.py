import numpy as np
from mysom.mysom import MySom
from pathlib import Path

x_path = Path('../../../../data/yvette_20_09_02/xwavehw.csv')
y_path = Path('../../../../data/yvette_20_11_18/shuffled_data_unnamed.csv')
figpath = Path('img/')
datestr = '2020_11_30'

x_data = np.genfromtxt(x_path, delimiter=',')
y_data = np.genfromtxt(y_path, delimiter=',')

label_list = ['']
marker_list = ['o']
colour_list = ['#FFA500']

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

# default som
som0 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=1)

# changing shape
som1 = MySom(x=11, y=8, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=1)

# changing sigma
som2 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=0.5, learning_rate=0.5, topology='rectangular', random_seed=1)
som3 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=2.0, learning_rate=0.5, topology='rectangular', random_seed=1)
som4 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=4.0, learning_rate=0.5, topology='rectangular', random_seed=1)

# changing learning rate
som5 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=1)
som6 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.3, topology='rectangular', random_seed=1)
som7 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.75, topology='rectangular', random_seed=1)

# changing sigma and learning rate
som8 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=1)
som9 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=2.0, topology='rectangular', random_seed=1)


# test parameter optimisation
som_list = [som0, som1, som2, som3, som4, som5, som6, som7, som8, som9]
for som in som_list:
    som.frobenius_norm(y_data)
    som.make_som(10000)
    som.make_labels(y_path, label_list, marker_list, colour_list)
    som.plot_som_umatrix(figpath, datestr)
    som.plot_som_scatter(figpath, datestr)
    som.plot_density_function(figpath, datestr)
    som.plot_neuron_activation_frequency(figpath, datestr)
    som.plot_errors(1000, figpath, datestr)
