import numpy as np
from mysom.mysom import MySom
from pathlib import Path
import datetime as dt

p = dt.datetime.now()
print("Start time is " + str(p))


x_path = Path('../../../../data/yvette_20_09_02/xwavehw.csv')
y_path = Path('../../../../data/yvette_20_09_02/High Wavenumbers for Dan.csv')
figpath = Path('img_som_plots/')
datestr = '2021_03_18'

x_data = np.genfromtxt(x_path, delimiter=',')
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 1057))

label_list = ['PNT2', 'LNCaP']
marker_list = ['o', 'x']
colour_list = ['#FFA500', '#FF00FF']

# 5 * sqrt(30) = 27.39

for i in range(1, 2):
    som0 = MySom(x=13, y=3, input_len=y_data.shape[1], sigma=1.6, learning_rate=0.9, topology='rectangular', random_seed=i)
    som0.som_setup(x_data, y_data, y_path, label_list, marker_list, colour_list)
    som0.frobenius_norm_normalisation()
    som0.train_som(10000)
    som0.plot_som_scatter(figpath, datestr, onlyshow=True)
    som0.plot_density_function(figpath, datestr, onlyshow=True)
