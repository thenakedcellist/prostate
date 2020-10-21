# %% setup

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import preprocessing
from minisom import MiniSom


# %% read files

# read data in csv format
xdata = np.genfromtxt('../data/banbury/_data_1-6/_wavenumbers.csv', delimiter=',', usecols=np.arange(1, 1016))
ydata = np.genfromtxt('../data/banbury/_data_1-6/all_data.csv', delimiter=',', usecols=np.arange(1, 1016))

# normalise data - sklearn method scaling to 0 mean and unit variance
sydata = preprocessing.scale(ydata)


# %% SOM train with rectangular topology

# initialise SOM with random weights and normalised data
som = MiniSom(16, 16, ydata.shape[1], sigma=1.0, learning_rate=0.1,
              neighborhood_function='gaussian', topology='rectangular',
              activation_distance='euclidean', random_seed=1)  # 5 * sqrt(2640) = 256.90
som.random_weights_init(sydata)

# train SOM and tell console that training is in progress
print('Training...')
som.train_random(sydata, 10000)  # number of iterations

# tell console training is complete
print('Training complete')


# %% setup label colours and markers

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
colors = ['r', 'g', 'b', 'c', 'm']  # edit marker colours


# %% plot distance map (u-matrix) and overlay mapped sample with uniform markers

# initialise figure canvas and single axes
fig1, ax1 = plt.subplots(1,1)
fig1.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)  # set whitespace around figure edges and space between subplots
fig1.suptitle("Self Organising Map of Porcine Eye Tissues", fontsize=16)

# fill in axes with SOM and overlaid data
ax1.pcolor(som.distance_map().T, cmap='Blues', alpha=1.0)  # plot transposed SOM distances in one matrix and set colormap
for cnt, xx in enumerate(sydata):
    bmu = som.winner(xx)  # calculate BMU for sample
    ax1.plot(bmu[0] + 0.5, bmu[1] + 0.5, markers[t[cnt]],
             markerfacecolor=colors[t[cnt]], markeredgecolor=colors[t[cnt]],
             markersize=4, markeredgewidth=1)  # place marker on winning SOM node for sample xx
ax1.axis([0, som._weights.shape[0], 0, som._weights.shape[1]])

# add colorbar to figure
divider1 = make_axes_locatable(ax1)
ax1_cb = divider1.new_horizontal(size=0.3, pad=0.1)
cb1 = colorbar.ColorbarBase(ax1_cb, cmap=cm.Blues, orientation='vertical', alpha=1.0)
cb1.ax.get_yaxis().labelpad = 16
cb1.ax.set_ylabel('distance from neurons in the neighbourhood',
                  rotation=270, fontsize=12)
fig1.add_axes(ax1_cb)

# add legend using proxy artists
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
ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.0, 1.08),
           borderaxespad=0, ncol=len(legend_elements), fontsize=10)

fig1.show()


# %% plot scatter plot of dots representing co-ordinates of winning neuron across map
# with random offset to avoid overlaps between points within same cell

# initialise figure canvas and single axes
fig2, ax2 = plt.subplots(1,1)
fig2.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)  # set whitespace around figure edges and space between subplots
fig2.suptitle("Scatter Plot of Self Organising Map", fontsize=16)

# fill in axes with SOM and overlaid scatter data
ax2.pcolor(som.distance_map().T, cmap='Blues', alpha=0.2)  # plot transposed SOM distances in one matrix and set colormap with reduced opacity
ax2.grid()  # print grid in bold over background
w_x, w_y = zip(*[som.winner(d) for d in sydata])  # get x an y variables
w_x = np.array(w_x)  # convert x variables into np array
w_y = np.array(w_y)  # convert y variables into np array
for c in np.unique(t):  # plot scatter plot for sample
    idx_t = t==c
    ax2.scatter(w_x[idx_t] + .5 + (np.random.rand(np.sum(idx_t)) - .5) * .5,
                w_y[idx_t] + .5 + (np.random.rand(np.sum(idx_t)) - .5) * .5, s=50, c=colors[c])

# add colorbar to figure
divider2 = make_axes_locatable(ax2)
ax2_cb = divider2.new_horizontal(size=0.3, pad=0.1)
cb2 = colorbar.ColorbarBase(ax2_cb, cmap=cm.Blues, orientation='vertical', alpha=0.2)
cb2.ax.get_yaxis().labelpad = 16
cb2.ax.set_ylabel('distance from neurons in the neighbourhood',
                  rotation=270, fontsize=12)
fig2.add_axes(ax2_cb)

# add legend using proxy artists
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
ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.0, 1.08),
           borderaxespad=0, ncol=len(legend_elements), fontsize=10)

plt.show()


# %% plot map showing frequency of neuron activation

# initialise figure canvas and single axes
fig3, ax3 = plt.subplots(1,1)
fig3.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)  # set whitespace around figure edges and space between subplots
fig3.suptitle("Frequency of Self organising Map Neuron Activation", fontsize=16)

# fill in axes with frequency of SOM neuron activation
frequencies = som.activation_response(sydata)  # generate frequency of neuron activation
ax3.pcolor(frequencies.T, cmap='Blues', alpha=1.0)  # plot tramsposed SOM frequencies in one matrix and set colourmap

# add colorbar to figure
divider3 = make_axes_locatable(ax3)
ax3_cb = divider3.new_horizontal(size=0.3, pad=0.1)
norm3 = mpl.colors.Normalize(vmin=np.min(frequencies), vmax=np.max(frequencies))  # define range for colorbar based on frequencies
cb3 = colorbar.ColorbarBase(ax=ax3_cb, cmap=cm.Blues, norm=norm3, alpha=1.0, orientation='vertical')
cb3.ax.get_yaxis().labelpad = 16
cb3.ax.set_ylabel('frequency of neuron activation',
                  rotation=270, fontsize=12)
fig3.add_axes(ax3_cb)

plt.show()
