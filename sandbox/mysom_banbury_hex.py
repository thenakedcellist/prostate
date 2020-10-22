# %% setup

from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


# %% read files

# read data in csv format
xdata = np.genfromtxt('../data/banbury/_data_1-6/_wavenumbers.csv', delimiter=',', usecols=np.arange(1, 1016))
ydata = np.genfromtxt('../data/banbury/_data_1-6/all_data.csv', delimiter=',', usecols=np.arange(1, 1016))

# normalise data to unity - sklearn method
sydata = preprocessing.scale(ydata)


# %% plot spectra

# data from yvette as they came
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
fig.suptitle('raw data')
ax1.plot(xdata, ydata[0])
ax2.plot(xdata, ydata[1])
ax3.plot(xdata, ydata[28])
ax4.plot(xdata, ydata[29])
plt.show()

# data normalised using sklearn
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
fig.suptitle('skearn normalised data')
ax1.plot(xdata, sydata[0])
ax2.plot(xdata, sydata[1])
ax3.plot(xdata, sydata[28])
ax4.plot(xdata, sydata[29])
plt.show()


# %% SOM train with hexagonal topology

# initialise SOM with random weights and normalised data
som = MiniSom(20, 20, ydata.shape[1], sigma=2.0, learning_rate=0.1,
              neighborhood_function='gaussian', topology='hexagonal',
              activation_distance='euclidean', random_seed=1)  # 5*sqrt(30) - 26
som.random_weights_init(sydata)

# train SOM and tell console that training is in progress
print('Training...')
som.train_random(sydata, 1000)  # number of iterations

# tell console training is complete
print('Training complete')


# %% setup colour and labels

# generate np array of colour labels from source data file
target = np.genfromtxt('../data/banbury/_data_1-6/all_data.csv', delimiter=',', usecols=(0), dtype=str)
# assign values to t given labels in input data with subdivisions (red1, red2, red3 etc.)
t = [0 if "cornea" in i else 1 if "lens" in i else 2 if "optic_nerve" in i else 3 if "retina" in i else 4 if "vitreous" in i else i for i in target]
np.array(t)
"""
# assign values to t given explicit labels in input data
t = np.zeros(len(target), dtype=int)
t[target == 'R'] = 0
t[target == 'G'] = 1
"""

# generate lists with assigned colours and markers for each label in t
markers = ['o', 'o', 'o', 'o', 'o']  # add markers to data
colors = ['r', 'r', 'r', 'r', 'r']  # edit marker colours

# %% plot distance map (u-matrix) and overlay mapped sample

from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D

# initiate figure canvas
f = plt.figure(figsize=(10,10))
ax = f.add_subplot(111)
ax.set_aspect('equal')  # sex axis ratio

# extract plotting data from SOM
xx, yy = som.get_euclidean_coordinates()
umatrix = som.distance_map()
weights = som.get_weights()

# create hex grid for SOM
for i in range(weights.shape[0]):
    for j in range(weights.shape[1]):
        wy = yy[(i, j)]*2/np.sqrt(3)*3/4
        hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95/np.sqrt(3),
                      facecolor=cm.Greys(umatrix[i, j]), alpha=.7, edgecolor='gray')
        ax.add_patch(hex)

# calculate and plot BMU for sample
for cnt, x in enumerate(sydata):
    w = som.winner(x)  # getting the winner
    # place a marker on the winning position for the sample xx
    wx, wy = som.convert_map_to_euclidean(w)
    wy = wy*2/np.sqrt(3)*3/4
    plt.plot(wx, wy, markers[t[cnt]-1], markerfacecolor=colors[t[cnt]-1],
             markeredgecolor=colors[t[cnt]-1], markersize=12, markeredgewidth=2)

# set range and label locations of axes
xrange = np.arange(weights.shape[0])
yrange = np.arange(weights.shape[1])
plt.xticks(xrange-.5, xrange)
plt.yticks(yrange*2/np.sqrt(3)*3/4, yrange)

# add colorbar to side of plot
divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Greys,
                            orientation='vertical', alpha=.4)
cb1.ax.get_yaxis().labelpad = 16
cb1.ax.set_ylabel('distance from neurons in the neighbourhood',
                  rotation=270, fontsize=16)
plt.gcf().add_axes(ax_cb)

# define legend
legend_elements = [Line2D([], [], marker=markers[0], color=colors[0], label='cornea',
                          markerfacecolor=colors[0], markersize=8, markeredgewidth=2,
                          linestyle='None', linewidth=0),
                   Line2D([], [], marker=markers[1], color=colors[1], label='lens',
                          markerfacecolor=colors[1], markersize=8, markeredgewidth=2,
                          linestyle='None', linewidth=0),
                   Line2D([], [], marker=markers[2], color=colors[2], label='optic nerve',
                          markerfacecolor=colors[2], markersize=8, markeredgewidth=2,
                          linestyle='None', linewidth=0),
                   Line2D([], [], marker=markers[3], color=colors[3], label='retina',
                          markerfacecolor=colors[3], markersize=8, markeredgewidth=2,
                          linestyle='None', linewidth=0),
                   Line2D([], [], marker=markers[4], color=colors[4], label='vitreous',
                          markerfacecolor=colors[4], markersize=8, markeredgewidth=2,
                          linestyle='None', linewidth=0)]
ax.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.08), loc='upper left',
          borderaxespad=0., ncol=3, fontsize=14)

plt.show()


# %%
"""
# %% plot scatter plot of dots representing co-ordinates of winning neuron across map
# with random offset to avoid overlaps between points within same cell
# TODO can this be overlaid on hex plot SOM?
# generate scatter plot data
w_x, w_y = zip(*[som.winner(d) for d in sydata])  # get list of x an y variables
w_x = np.array(w_x)  # convert x variables into np array
w_y = np.array(w_y)  # convert y variables into np array

# initialise figure canvas with SOM
plt.figure()
plt.pcolor(som.distance_map().T, cmap='Blues', alpha=.2)  # plot SOM distances in one matrix, transpose distances using .T, and set colourmap with reduced opacity
plt.colorbar()  # add legend of normalised values
plt.grid()  # print grid in bold over background

# plot scatter plot for sample
for c in np.unique(t):
    idx_t = t==c
    plt.scatter(w_x[idx_t]+.5+(np.random.rand(np.sum(idx_t))-.5)*.8,
                w_y[idx_t]+.5+(np.random.rand(np.sum(idx_t))-.5)*.8, s=50, c=colors[c])

plt.show()
"""