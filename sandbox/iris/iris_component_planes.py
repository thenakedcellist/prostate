import numpy as np
from mysom.mysom import MySom
from pathlib import Path

x_path = Path('../../../data/practice_datasets/iris_dataset/iris.txt')
y_path = Path('../../../data/practice_datasets/iris_dataset/iris.txt')
figpath = Path('img/')
datestr = '2021_01_20'

x_data = np.genfromtxt(x_path, delimiter=',', usecols=0, dtype='str')
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 5))
label_list = ['setosa', 'versicolor', 'virginica']
marker_list = ['o', 'x', '_']
colour_list = ['#FFA500', '#FFA500', '#FFA500']

# 5*SQRT(150) = 61.24 so 8 . 8 grid

som = MySom(x=8, y=8, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=1)
som.frobenius_norm_normalisation(y_data)
som.make_som(10000)
som.make_labels(y_path, label_list, marker_list, colour_list)
som.plot_som_umatrix(figpath, datestr, onlyshow=True)
som.plot_som_scatter(figpath, datestr, onlyshow=True)
som.plot_density_function(figpath, datestr, onlyshow=True)


# from MiniSom
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv(Path('../../../data/practice_datasets/iris_dataset/iris.txt'), delimiter=',',
                 names=['species', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
target = df.iloc[:, 0]
Features = df.iloc[:, 1:]

feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
feat_num = len(feature_names)
corrMatrix = df.corr().round(2)
sn.heatmap(corrMatrix, annot=True)
plt.show()

W = som.get_weights()
plt.figure()
for i, v in enumerate(feature_names):
    plt.subplot(2, 2, i + 1)
    plt.title(v)
    plt.pcolor(W[:, :, i].T, cmap='coolwarm')
    plt.xticks(np.arange(som.x + 1))
    plt.yticks(np.arange(som.y + 1))
plt.tight_layout()
plt.show()

# sepal width appears to be negatively correlated with category - what happens if it is removed
dim_y_data = np.delete(y_data, 1, 1)
dim_som = MySom(x=8, y=8, input_len=dim_y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=1)
dim_som.frobenius_norm_normalisation(dim_y_data)
dim_som.make_som(10000)
dim_som.make_labels(y_path, label_list, marker_list, colour_list)
dim_som.plot_som_umatrix(figpath, datestr, onlyshow=True)
dim_som.plot_som_scatter(figpath, datestr, onlyshow=True)
dim_som.plot_density_function(figpath, datestr, onlyshow=True)

dim_df = pd.read_csv(Path('../../../data/practice_datasets/iris_dataset/iris.txt'), delimiter=',',
                 names=['species', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
dim_target = df.iloc[:, 0]
dim_Features = df.iloc[:, [1, 3, 4]]

dim_feature_names = ['sepal_length', 'petal_length', 'petal_width']
dim_feat_num = len(dim_feature_names)
corrMatrix = dim_Features.corr().round(2)
sn.heatmap(corrMatrix, annot=True)
plt.show()

W = som.get_weights()
plt.figure()
for i, v in enumerate(dim_feature_names):
    plt.subplot(2, 2, i + 1)
    plt.title(v)
    plt.pcolor(W[:, :, i].T, cmap='Blues_r')
    plt.xticks(np.arange(som.x + 1))
    plt.yticks(np.arange(som.y + 1))
plt.tight_layout()
plt.show()

# TODO continue https://github.com/JustGlowing/minisom/blob/master/examples/FeatureSelection.ipynb