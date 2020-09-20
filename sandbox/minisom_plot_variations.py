# %% setup

from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt


# %% read data

# read data in csv format
data = np.genfromtxt(r'H:\MSc\project\practice_datasets\colours_dataset_small\colours_edit_secondary.csv', delimiter=',', skip_header=1, usecols=(0, 1, 2))


# normalise data to unity
ndata = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)


# %% SOM training

# initialise SOM with random weights and normalised data
som = MiniSom(7, 7, 3, sigma=1.0, learning_rate=0.1, topology='rectangular', random_seed=1)  # topology can be hexagonal for hex plot
som.random_weights_init(ndata)

# train SOM and tell console that training is in progress
print('Training...')
som.train_random(ndata, 1000)  # number of iterations

# tell console training is complete
print('Training complete')


# %% setup colour and labels

# load colour labels
target = np.genfromtxt(r'H:\MSc\project\practice_datasets\colours_dataset_small\colours_edit_secondary.csv', delimiter=',', skip_header=1, usecols=(3), dtype=str)
t = np.zeros(len(target), dtype=int)
t[target == 'R'] = 0
t[target == 'G'] = 1
t[target == 'B'] = 2
t[target == 'C'] = 3
t[target == 'M'] = 4
t[target == 'Y'] = 5
t[target == 'K'] = 6

# assign colours and markers to each label
markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o']  # add markers to data
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # edit marker colours


# %% plot distance map (u-matrix) and overlay mapped sample

# initialise figure canvas with SOM
plt.figure(figsize=((24/2.54), (18/2.52)))
plt.pcolor(som.distance_map().T, cmap='Greys')  # plot SOM distances in one matrix, transpose distances using .T, and set colourmap
plt.colorbar()  # add legend of normalised values

# calculate and plot BMU for sample
for cnt, xx in enumerate(ndata):
    bmu = som.winner(xx)  # calculate BMU
    plt.plot(bmu[0]+.5, bmu[1]+.5, markers[t[cnt]], markerfacecolor=colors[t[cnt]], markeredgecolor=colors[t[cnt]], markersize=6, markeredgewidth=2)  # place marker on winning position for sample xx
plt.axis([0, som._weights.shape[0], 0, som._weights.shape[1]])

plt.show()


# %% plot scatter plot of dots representing co-ordinates of winning neuron across map with random offset to avoid overlaps between pojnts within same cell

# generate scatter plot data
w_x, w_y = zip(*[som.winner(d) for d in ndata])  # get x an y variables
w_x = np.array(w_x)  # convert x variables into np array
w_y = np.array(w_y)  # convert y variables into np array

# initialise figure canvas with SOM
plt.figure(figsize=((24/2.54), (18/2.52)))
plt.pcolor(som.distance_map().T, cmap='Greys', alpha=.2)  # plot SOM distances in one matrix, transpose distances using .T, and set colourmap with reduced opacity
plt.colorbar()  # add legend of normalised values
plt.grid()  # print grid in bold over background

# plot scatter plot for sample
for c in np.unique(t):
    idx_t = t==c
    plt.scatter(w_x[idx_t]+.5+(np.random.rand(np.sum(idx_t))-.5)*.8, w_y[idx_t]+.5+(np.random.rand(np.sum(idx_t))-.5)*.8, s=50, c=colors[c])

plt.show()


# %% plot map showing frequency of neuron activation

# initialise figure canvas
plt.figure(figsize=((24/2.54), (18/2.52)))
frequencies = som.activation_response(ndata)  # generate frequency of neuron activation
plt.pcolor(frequencies.T, cmap='Greys')  # plot SOM frequencies in one matrix, transpose distances using .T, and set colourmap
plt.colorbar()  # add legend of normalised values

plt.show()


# %% plot quantisation and topographic error of SOM at each step
# this helps to understnad training and to estimate number of iterations to run

# define iteration bounds and declare errors
max_iter= 10000
q_error = []
t_error = []

# calculate errors for each iteration of SOM
for i in range(max_iter):
    rand_i = np.random.randint(len(ndata))
    som.update(ndata[rand_i], som.winner(ndata[rand_i]), i, max_iter)
    q_error.append(som.quantization_error(ndata))
    t_error.append(som.topographic_error(ndata))

# initialise figure canvas
plt.plot(np.arange(max_iter), q_error, label='quantisation error')
plt.plot(np.arange(max_iter), t_error, label='topographic error')
plt.ylabel('quantisation error')
plt.xlabel('iteration index')
plt.legend()

plt.show()


# %% plot hexagonal SOM

from matplotlib.patches import RegularPolygon, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D

# initialise figure canvas
f = plt.figure(figsize=(10, 10))
ax = f.add_subplot(111)
ax.set_aspect('equal')

# get size, values, and weights from SOM
xx, yy = som.get_euclidean_coordinates()
umatrix = som.distance_map()
weights = som.get_weights()

# form hex grid
for i in range(weights.shape[0]):
    for j in range(weights.shape[1]):
        wy = yy[(i, j)]*2/np.sqrt(3)*3/4
        hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95/np.sqrt(3), facecolor=cm.Greys(umatrix[i, j]), alpha=.4, edgecolor='gray')
        ax.add_patch(hex)

# calculate and plot BMU for sample
for cnt, x in enumerate(ndata):
    bmu = som.winner(x)  # calculate BMU
    wx, wy = som.convert_map_to_euclidean(bmu)
    wy = wy*2/np.sqrt(3)*3/4
    plt.plot(wx, wy, markers[t[cnt]], markerfacecolor=colors[t[cnt]], markeredgecolor=colors[t[cnt]], markersize=6, markeredgewidth=2)  # place marker on winning position for sample xx

# set x and y range of plot
xrange = np.arange(weights.shape[0])
yrange = np.arange(weights.shape[1])
plt.xticks(xrange-.5, xrange)
plt.yticks(yrange*2/np.sqrt(3)*3/4, yrange)

# set second axes of plot
divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal( size="5%", pad=0.05)
cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Greys, orientation='vertical', alpha=.4)
cb1.ax.get_yaxis().labelpad = 16
cb1.ax.set_ylabel('distance from neurons in the neighbourhood', rotation=270, fontsize=16)
plt.gcf().add_axes(ax_cb)

plt.show()
