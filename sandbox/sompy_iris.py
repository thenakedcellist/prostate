# %% setup and imports

import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from time import time
import sompy


# %% import data

# read iris database in csv format
data = np.genfromtxt('iris.csv', delimiter=',', usecols=(0, 1, 2, 3))

# normalisation of dataset to unity
data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)


# %% plotting data

fig = plt.figure()
plt.plot(data[:, 0], data[:, 1], 'ob', alpha=0.2, markersize=4)
fig.set_size_inches(10, 10)


# %% set up map and train

# set size of map
mapsize = [20, 20]

# default parameters for training
som = sompy.SOMFactory.build(data, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy')

# initialise training
som.train(n_job=1, verbose='info')


# %% plot Kohonen map

# set component names
som.component_names = ['1', '2', '3', '4']

# create Kohonen map
v = sompy.mapview.View2DPacked(50, 50, 'test', text_size=8)

# visualise map with given settings
v.show(som, what='cluster',  # 'codebook' for values 'cluster' for group
       which_dim='all',  # all dimensions or select number
       cmap='jet',  # style of colour
       col_sz=6)  # set column size


# %% show clustering of data
"""
c1 = som.cluster(n_clusters=3)  # set number of clusters
getattr(som, 'cluster_labels')  # extract cluster lables from SOM
"""

# %% view cluster map with overlaid cluster values

h = sompy.hitmap.HitMapView(10, 10, 'hitmap', text_size=8, show_text=True)
h.show(som)


# %% 