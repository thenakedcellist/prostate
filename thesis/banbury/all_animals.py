import numpy as np
from mysom.mysom import MySom
from pathlib import Path

x_path = Path('../../../data/banbury/_all_animals/all_animal_data.csv')
y_path = Path('../../../data/banbury/_all_animals/all_animal_data.csv')
figpath = Path('img_all_animals/')
datestr = '2021_02_12'

x_data = np.genfromtxt(x_path, delimiter=',', usecols=np.arange(1, 1016))
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 1016))

label_list = ['cornea', 'lens', 'optic_nerve', 'retina', 'vitreous']  # default is 'Blinded Data'
marker_list = ['o', '_', 'x', '|', 'd']
colour_list = ['#20ABC4', '#2FAB40', '#101010', '#640098', '#E66510']

# Babnury group used 16 . 16 SOM

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

som12 = MySom(x=16, y=16, input_len=y_data.shape[1], sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=1)

# initiate and plot each SOM
som_list = [som12]
for som in som_list:
    som.som_setup(x_data, y_data, y_path, label_list, marker_list, colour_list)
    som.scikit_normalisation()
    som.train_som(10000)
    som.plot_som_umatrix(figpath, datestr)
    som.plot_som_scatter(figpath, datestr)
    som.plot_density_function(figpath, datestr)
    som.plot_node_activation_frequency(figpath, datestr)
    #som.plot_errors(1000, figpath, datestr)
