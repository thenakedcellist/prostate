import numpy as np
from mysom.mysom import MySom
from pathlib import Path

x_path = Path('../../../data/banbury/_data_1-6/_wavenumbers.csv')
y_path = Path('../../../data/banbury/_data_1-6/all_data.csv')
figpath = Path('img_1-6/')
datestr = '2021_02_11'

x_data = np.genfromtxt(x_path, delimiter=',', usecols=np.arange(1, 1016))
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 1016))

label_list = ['cornea', 'lens', 'optic_nerve', 'retina', 'vitreous']  # default is 'Blinded Data'
marker_list = ['o', '_', 'x', '|', 'd']
colour_list = ['#20ABC4', '#2FAB40', '#101010', '#640098', '#E66510']

# 5*SQRT(2640) = 256.90 so 16 . 16 SOM

'''
Default values
--------------
x by y: 5*SQRT(n)
sigma:  1.0
learning rate:  0.5
random seed:    1
'''

som0 = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=1)
som1 = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=2.0, learning_rate=0.5, topology='rectangular', random_seed=1)
som2 = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=3.0, learning_rate=0.5, topology='rectangular', random_seed=1)
som3 = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=4.0, learning_rate=0.5, topology='rectangular', random_seed=1)
som4 = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=1.0, learning_rate=1.0, topology='rectangular', random_seed=1)
som5 = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=2.0, learning_rate=1.0, topology='rectangular', random_seed=1)
som6 = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=1)
som7 = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=4.0, learning_rate=1.0, topology='rectangular', random_seed=1)
som8 = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=1)
som9 = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=2.0, learning_rate=0.5, topology='rectangular', random_seed=1)
som10 = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=3.0, learning_rate=0.5, topology='rectangular', random_seed=1)
som11 = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=4.0, learning_rate=0.5, topology='rectangular', random_seed=1)
som_banbury_parameters = [som0, som1, som2, som3, som4, som5, som6, som7, som8, som9, som10, som11]

som12a = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=1)
som12b = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=2)
som12c = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=3)
som12d = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=4)
som12e = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=5)
som12f = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=6)
som12g = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=7)
som12h = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=8)
som12i = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=9)
som12j = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=10)
som12 = [som12a, som12b, som12c, som12d, som12e, som12f, som12g, som12h, som12i, som12j]

# initiate and plot each SOM
som_list = [som12]
for soms in som_list:
    for som in soms:
        som.som_setup(x_data, y_data, y_path, label_list, marker_list, colour_list)
        som.scikit_normalisation()
        som.train_som(10000)
        som.plot_som_umatrix(figpath, datestr)
        som.plot_som_scatter(figpath, datestr)
        som.plot_density_function(figpath, datestr)
        som.plot_node_activation_frequency(figpath, datestr)
        som.plot_errors(1000, figpath, datestr)
