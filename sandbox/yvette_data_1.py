# %% setup

from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn import preprocessing


# %% read files

# read data in csv format
xdata = np.genfromtxt("data/yvette_02_09_20/xwavehw.csv", delimiter=',')
ydata = np.genfromtxt("data/yvette_02_09_20/High Wavenumbers for Dan.csv", delimiter=',', usecols=np.arange(1, 1057))

# normalise data - MiniSom method dividing each column by Frobenius norm
nydata = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, ydata)

# normalise data - sklearn method scaling to 0 mean and unit variance
sydata = preprocessing.scale(ydata.T)


# %% SOM train with rectangular topology

# initialise SOM with random weights and normalised data
som = MiniSom(6, 5, ydata.shape[1], sigma=1.0, learning_rate=0.1,
              neighborhood_function='gaussian', topology='rectangular',
              activation_distance='euclidean', random_seed=1)  # 5*sqrt(30) - 26
som.random_weights_init(nydata)

# train SOM and tell console that training is in progress
print('Training...')
som.train_random(nydata, 1000)  # number of iterations

# tell console training is complete
print('Training complete')


# %% setup colour and labels

# generate np array of colour labels from source data file
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

# generate lists with assigned colours and markers for each label in t
markers = ['o', 'o']  # add markers to data
colors = ['r', 'g']  # edit marker colours


# %% plot distance map (u-matrix) and overlay mapped sample

# initialise figure canvas with SOM
fig1, ax1 = plt.subplots(1,1,figsize=(9,6))
ax1.pcolor(som.distance_map().T, cmap='Blues')  # plot SOM distances in one matrix, transpose distances using .T, and set colourmap
cax = ax1.imshow(som.distance_map().T, cmap='Blues')  # create mappable of colourmap to define colorbar
fig1.colorbar(cax, ax=ax1)  # add legend of normalised values
fig1.suptitle("Self Organising Map of PNT2 and LNCaP Cell Lines", fontsize=16)


# calculate and plot BMU for sample
for cnt, xx in enumerate(nydata):
    bmu = som.winner(xx)  # calculate BMU
    plt.plot(bmu[0] + .5, bmu[1] + .5, markers[t[cnt]], markerfacecolor=colors[t[cnt]], markeredgecolor=colors[t[cnt]],
         markersize=6, markeredgewidth=2)  # place marker on winning position for sample xx
ax1.axis([0, som._weights.shape[0], 0, som._weights.shape[1]])

# define legend using proxy artists
legend_elements = [Line2D([], [], marker='o', color=colors[0], label='PNT2',
                       markerfacecolor=colors[0], markersize=12, markeredgewidth=2,
                       linewidth=0, linestyle='None'),
                   Line2D([], [], marker='o', color=colors[1], label='LNCaP',
                       markerfacecolor=colors[1], markersize=12, markeredgewidth=2,
                       linewidth=0, linestyle='None')]
ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.05, 1.1),
           borderaxespad=0, ncol=len(legend_elements), fontsize=10)


fig1.show()
''' from hex-plot legend
# add colorbar to side of plot
divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Greys,
                            orientation='vertical', alpha=.4)
cb1.ax.get_yaxis().labelpad = 16
cb1.ax.set_ylabel('distance from neurons in the neighbourhood',
                  rotation=270, fontsize=16)
plt.gcf().add_axes(ax_cb)
'''


# %% plot scatter plot of dots representing co-ordinates of winning neuron across map
# with random offset to avoid overlaps between points within same cell

# generate np arrays containing scatter plot data
w_x, w_y = zip(*[som.winner(d) for d in nydata])  # get x an y variables
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
    plt.scatter(w_x[idx_t] + .5 + (np.random.rand(np.sum(idx_t)) - .5) * .8,
                w_y[idx_t] + .5 + (np.random.rand(np.sum(idx_t)) - .5) * .8, s=50, c=colors[c])

plt.show()


# %% plot map showing frequency of neuron activation

# initialise figure canvas
plt.figure()
frequencies = som.activation_response(nydata)  # generate frequency of neuron activation
plt.pcolor(frequencies.T, cmap='Blues')  # plot SOM frequencies in one matrix, transpose distances using .T, and set colourmap
plt.colorbar()  # add legend of normalised values

plt.show()


# %% plot quantisation and topographic error of SOM at each step
# this helps to understand training and to estimate number of iterations to run

# define iteration bounds and declare error lists
max_iter= 10000
q_error = []
t_error = []

# tell console training is in progress
print('error calculation...')

# calculate errors for each iteration of SOM and append to error lists
for i in range(max_iter):
    rand_i = np.random.randint(len(nydata))
    som.update(nydata[rand_i], som.winner(nydata[rand_i]), i, max_iter)
    q_error.append(som.quantization_error(nydata))
    t_error.append(som.topographic_error(nydata))

# tell console error calculation is complete
print('error calculation complete')

# initialise figure canvas
plt.plot(np.arange(max_iter), q_error, label='quantisation error')
plt.plot(np.arange(max_iter), t_error, label='topographic error')
plt.ylabel('quantisation error')
plt.xlabel('iteration index')
plt.legend()

plt.show()

# %% todo list
# TODO define explicit fig, ax and then utilise ax.fn() to ensure I know exactly what is being manipulated
# TODO figsize to fit everyhting neatly
# TODO give each plot a title and axes legends
# TODO make SOM u-matrix marker size proportional to number of vectors activating that neuron
# TODO refine separate steps into functions to help with structure and modularisation
# TODO form script into module
