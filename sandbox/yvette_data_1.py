# %% setup

from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


# %% read files

# read data in csv format
xdata = np.genfromtxt("data/yvette_02_09_20/xwavehw.csv", delimiter=',')
ydata = np.genfromtxt("data/yvette_02_09_20/High Wavenumbers for Dan.csv", delimiter=',', usecols=np.arange(1, 1057))

# normalise data to unity - numpy method
nydata = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, ydata)

# normalise data to unity - sklearn method
sydata = preprocessing.scale(ydata.T)


# %% plot spectra

# data from yvette as they came
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
fig.suptitle('cleaned data')
ax1.plot(xdata, ydata[0])
ax2.plot(xdata, ydata[1])
ax3.plot(xdata, ydata[28])
ax4.plot(xdata, ydata[29])
plt.show()

# data normalised using numpy
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
fig.suptitle('numpy normalised cleaned data')
ax1.plot(xdata, nydata[0])
ax2.plot(xdata, nydata[1])
ax3.plot(xdata, nydata[28])
ax4.plot(xdata, nydata[29])
plt.show()

# data normalised using sklearn
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
fig.suptitle('skearn normalised cleaned data')
ax1.plot(xdata, sydata.T[0])
ax2.plot(xdata, sydata.T[1])
ax3.plot(xdata, sydata.T[28])
ax4.plot(xdata, sydata.T[29])
plt.show()


# %% SOM train with hexagonal topology

# initialise SOM with random weights and normalised data
som = MiniSom(6, 5, ydata.shape[1], sigma=1.0, learning_rate=0.1,
              neighborhood_function='gaussian', topology='hexagonal',
              activation_distance='euclidean', random_seed=1)  # 5*sqrt(30) - 26
som.random_weights_init(nydata)

# train SOM and tell console that training is in progress
print('Training...')
som.train_random(nydata, 1000)  # number of iterations

# tell console training is complete
print('Training complete')


# %% setup colour and labels

# load colour labels from source data file
target = np.genfromtxt(r"data/yvette_02_09_20/High Wavenumbers for Dan.csv", delimiter=',', usecols=(0), dtype=str)
# assign values to t given labels in input data with subdivisions (red1, red2, red3 etc.)
t = [0 if "PNT2" in i else 1 if "LNCaP" in i else i for i in target]
np.array(t)
"""
# assign values to t given explicit labels in input data
t = np.zeros(len(target), dtype=int)
t[target == 'R'] = 0
t[target == 'G'] = 1
"""

# assign colours and markers to each label
markers = ['o', 'o']  # add markers to data
colors = ['r', 'g']  # edit marker colours


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
for cnt, x in enumerate(nydata):
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
legend_elements = [Line2D([0], [0], marker='o', color='r', label='PNT2',
                   markerfacecolor='r', markersize=14, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='o', color='g', label='LNCaP',
                   markerfacecolor='g', markersize=14, linestyle='None', markeredgewidth=2)]
ax.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.08), loc='upper left',
          borderaxespad=0., ncol=3, fontsize=14)

plt.show()


# %% plot scatter plot of dots representing co-ordinates of winning neuron across map
# with random offset to avoid overlaps between points within same cell
# TODO can this be overlaid on hex plot SOM?
# generate scatter plot data
w_x, w_y = zip(*[som.winner(d) for d in nydata])  # get list of x an y variables
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



# %% plot map showing frequency of neuron activation

# initialise figure canvas
plt.figure()
frequencies = som.activation_response(nydata)  # generate frequency of neuron activation
plt.pcolor(frequencies.T, cmap='Blues')  # plot SOM frequencies in one matrix, transpose distances using .T, and set colourmap
plt.colorbar()  # add legend of normalised values

plt.show()


# %% plot quantisation error of SOM at each step
# this helps to understand training and to estimate number of iterations to run
# topographic error is not implemented for hexagonal topology

# define iteration bounds and declare errors
max_iter= 10000
q_error = []

# tell console training is in progress
print('error calculation...')

# calculate errors for each iteration of SOM
for i in range(max_iter):
    rand_i = np.random.randint(len(nydata))
    som.update(nydata[rand_i], som.winner(nydata[rand_i]), i, max_iter)
    q_error.append(som.quantization_error(nydata))

# tell console error calculation is complete
print('error calculation complete')

# initialise figure canvas
plt.plot(np.arange(max_iter), q_error, label='quantisation error')
plt.ylabel('quantisation error')
plt.xlabel('iteration index')
plt.legend()

plt.show()
