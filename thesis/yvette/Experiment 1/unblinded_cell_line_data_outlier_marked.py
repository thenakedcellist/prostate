import numpy as np
from mysom.mysom import MySom
from pathlib import Path

x_path = Path('../../../../data/yvette_20_09_02/xwavehw.csv')
y_path = Path('../../../../data/yvette_20_11_18/shuffled_data_named.csv')
figpath = Path('img_unblinded_cell_line_data_outlier_marked/')
datestr = '2020_12_17'

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

# default som
somA1 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=1)
somA2 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=2)
somA3 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=3)
somA4 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=4)
somA5 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=5)
somA6 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=6)
somA7 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=7)
somA8 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=8)
somA9 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=9)
somA10 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=10)
somA = [somA1, somA2, somA3, somA4, somA5, somA6, somA7, somA8, somA9, somA10]

# changing shape
somB1 = MySom(x=11, y=8, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=1)
somB2 = MySom(x=11, y=8, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=2)
somB3 = MySom(x=11, y=8, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=3)
somB4 = MySom(x=11, y=8, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=4)
somB5 = MySom(x=11, y=8, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=5)
somB6 = MySom(x=11, y=8, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=6)
somB7 = MySom(x=11, y=8, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=7)
somB8 = MySom(x=11, y=8, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=8)
somB9 = MySom(x=11, y=8, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=9)
somB10 = MySom(x=11, y=8, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=10)
somB = [somB1, somB2, somB3, somB4, somB5, somB6, somB7, somB8, somB9, somB10]

# changing sigma
somC1 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=0.5, learning_rate=0.5, topology='rectangular', random_seed=1)
somC2 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=0.5, learning_rate=0.5, topology='rectangular', random_seed=2)
somC3 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=0.5, learning_rate=0.5, topology='rectangular', random_seed=3)
somC4 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=0.5, learning_rate=0.5, topology='rectangular', random_seed=4)
somC5 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=0.5, learning_rate=0.5, topology='rectangular', random_seed=5)
somC6 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=0.5, learning_rate=0.5, topology='rectangular', random_seed=6)
somC7 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=0.5, learning_rate=0.5, topology='rectangular', random_seed=7)
somC8 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=0.5, learning_rate=0.5, topology='rectangular', random_seed=8)
somC9 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=0.5, learning_rate=0.5, topology='rectangular', random_seed=9)
somC10 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=0.5, learning_rate=0.5, topology='rectangular', random_seed=10)
somC = [somC1, somC2, somC3, somC4, somC5, somC6, somC7, somC8, somC9, somC10]

somD1 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=2.0, learning_rate=0.5, topology='rectangular', random_seed=1)
somD2 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=2.0, learning_rate=0.5, topology='rectangular', random_seed=2)
somD3 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=2.0, learning_rate=0.5, topology='rectangular', random_seed=3)
somD4 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=2.0, learning_rate=0.5, topology='rectangular', random_seed=4)
somD5 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=2.0, learning_rate=0.5, topology='rectangular', random_seed=5)
somD6 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=2.0, learning_rate=0.5, topology='rectangular', random_seed=6)
somD7 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=2.0, learning_rate=0.5, topology='rectangular', random_seed=7)
somD8 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=2.0, learning_rate=0.5, topology='rectangular', random_seed=8)
somD9 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=2.0, learning_rate=0.5, topology='rectangular', random_seed=9)
somD10 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=2.0, learning_rate=0.5, topology='rectangular', random_seed=10)
somD = [somD1, somD2, somD3, somD4, somD5, somD6, somD7, somD8, somD9, somD10]

somE1 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=4.0, learning_rate=0.5, topology='rectangular', random_seed=1)
somE2 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=4.0, learning_rate=0.5, topology='rectangular', random_seed=2)
somE3 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=4.0, learning_rate=0.5, topology='rectangular', random_seed=3)
somE4 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=4.0, learning_rate=0.5, topology='rectangular', random_seed=4)
somE5 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=4.0, learning_rate=0.5, topology='rectangular', random_seed=5)
somE6 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=4.0, learning_rate=0.5, topology='rectangular', random_seed=6)
somE7 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=4.0, learning_rate=0.5, topology='rectangular', random_seed=7)
somE8 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=4.0, learning_rate=0.5, topology='rectangular', random_seed=8)
somE9 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=4.0, learning_rate=0.5, topology='rectangular', random_seed=9)
somE10 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=4.0, learning_rate=0.5, topology='rectangular', random_seed=10)
somE = [somE1, somE2, somE3, somE4, somE5, somE6, somE7, somE8, somE9, somE10]

# changing learning rate
somF1 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=1)
somF2 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=2)
somF3 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=3)
somF4 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=4)
somF5 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=5)
somF6 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=6)
somF7 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=7)
somF8 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=8)
somF9 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=9)
somF10 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=10)
somF = [somF1, somF2, somF3, somF4, somF5, somF6, somF7, somF8, somF9, somF10]

somG1 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.3, topology='rectangular', random_seed=1)
somG2 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.3, topology='rectangular', random_seed=2)
somG3 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.3, topology='rectangular', random_seed=3)
somG4 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.3, topology='rectangular', random_seed=4)
somG5 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.3, topology='rectangular', random_seed=5)
somG6 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.3, topology='rectangular', random_seed=6)
somG7 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.3, topology='rectangular', random_seed=7)
somG8 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.3, topology='rectangular', random_seed=8)
somG9 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.3, topology='rectangular', random_seed=9)
somG10 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.3, topology='rectangular', random_seed=10)
somG = [somG1, somG2, somG3, somG4, somG5, somG6, somG7, somG8, somG9, somG10]

somH1 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.75, topology='rectangular', random_seed=1)
somH2 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.75, topology='rectangular', random_seed=2)
somH3 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.75, topology='rectangular', random_seed=3)
somH4 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.75, topology='rectangular', random_seed=4)
somH5 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.75, topology='rectangular', random_seed=5)
somH6 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.75, topology='rectangular', random_seed=6)
somH7 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.75, topology='rectangular', random_seed=7)
somH8 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.75, topology='rectangular', random_seed=8)
somH9 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.75, topology='rectangular', random_seed=9)
somH10 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.75, topology='rectangular', random_seed=10)
somH = [somH1, somH2, somH3, somH4, somH5, somH6, somH7, somH8, somH9, somH10]

# changing sigma and learning rate
somI1 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=1)
somI2 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=2)
somI3 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=3)
somI4 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=4)
somI5 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=5)
somI6 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=6)
somI7 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=7)
somI8 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=8)
somI9 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=9)
somI10 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=10)
somI = [somI1, somI2, somI3, somI4, somI5, somI6, somI7, somI8, somI9, somI10]

somJ1 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=2.0, topology='rectangular', random_seed=1)
somJ2 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=2.0, topology='rectangular', random_seed=2)
somJ3 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=2.0, topology='rectangular', random_seed=3)
somJ4 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=2.0, topology='rectangular', random_seed=4)
somJ5 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=2.0, topology='rectangular', random_seed=5)
somJ6 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=2.0, topology='rectangular', random_seed=6)
somJ7 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=2.0, topology='rectangular', random_seed=7)
somJ8 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=2.0, topology='rectangular', random_seed=8)
somJ9 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=2.0, topology='rectangular', random_seed=9)
somJ10 = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=2.0, topology='rectangular', random_seed=10)
somJ = [somJ1, somJ2, somJ3, somJ4, somJ5, somJ6, somJ7, somJ8, somJ9, somJ10]

# test parameter optimisation
som_list = [somA] #, somB, somC, somD, somE, somF, somG, somH, somI, somJ]
for soms in som_list:
    for som in soms:
        som.frobenius_norm_normalisation(y_data)
        som.make_som(10000)
        som.make_labels(y_path, label_list, marker_list, colour_list)
        som.plot_som_umatrix(figpath, datestr)
        som.plot_som_scatter(figpath, datestr)
        som.plot_density_function(figpath, datestr)
        som.plot_neuron_activation_frequency(figpath, datestr)
        som.plot_errors(1000, figpath, datestr)
