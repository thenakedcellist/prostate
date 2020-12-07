import numpy as np
from mysom.mysom import MySom
from pathlib import Path

x_path = Path('../../../data/practice_datasets/iris_dataset/iris.txt')
y_path = Path('../../../data/practice_datasets/iris_dataset/iris.txt')
figpath = Path('img/')
datestr = '2020_12_03'

x_data = np.genfromtxt(x_path, delimiter=',')
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 4))

label_list = ['setosa', 'versicolor', 'virginica']  # default is 'Blinded Data'
marker_list = ['o', 'x', '_']
colour_list = ['#FFA500', '#FFA500', '#FFA500']

# 5*SQRT(150) = 61.24 so 8 . 8 grid

'''
Default values
--------------
x by y: 5*SQRT(n)
sigma:  1.0
learning rate:  0.5
random seed:    1
'''

som0 = MySom(x=8, y=8, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=1)
som1 = MySom(x=10, y=7, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=1)
som2 = MySom(x=8, y=8, input_len=y_data.shape[1], sigma=5.0, learning_rate=1.0, topology='rectangular', random_seed=1)
som3 = MySom(x=8, y=8, input_len=y_data.shape[1], sigma=3.0, learning_rate=0.5, topology='rectangular', random_seed=1)
som4 = MySom(x=8, y=8, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=42)
som5 = MySom(x=8, y=8, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=2020)
som6 = MySom(x=8, y=8, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=5141)

# initiate and plot each SOM
som_list = [som4, som5, som6]
for som in som_list:
    som.frobenius_norm(y_data)
    som.make_som(10000)
    som.make_labels(y_path, label_list, marker_list, colour_list)
    som.plot_som_umatrix(figpath, datestr)
    som.plot_som_scatter(figpath, datestr)
    som.plot_density_function(figpath, datestr)
    som.plot_neuron_activation_frequency(figpath, datestr)
    som.plot_errors(1000, figpath, datestr)
