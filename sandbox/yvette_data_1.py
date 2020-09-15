# %% setup

from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


# %% read files

# read data in csv format
xdata = np.genfromtxt(r'H:\MSc\project\data\yvette_02-09-20\xwavehw.csv', delimiter=',')
ydata = np.genfromtxt(r'H:\MSc\project\data\yvette_02-09-20\High Wavenumbers for Dan.csv', delimiter=',', usecols=np.arange(1, 1057))

# normalise data to unity - numpy method
nydata = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, ydata)

# normalise data to unity - sklearn method
sydata = preprocessing.scale(ydata)


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
ax1.plot(xdata, sydata[0])
ax2.plot(xdata, sydata[1])
ax3.plot(xdata, sydata[28])
ax4.plot(xdata, sydata[29])
plt.show()



# %% SOM train#

# initialise SOM with random weights and normalised data
som = MiniSom(5, 5, 1056, sigma=1.0, learning_rate=0.1, random_seed=1)  # 5*sqrt30 - 26
som.random_weights_init(nydata)

# train SOM and tell console that training is in progress
print('Training...')
som.train_random(nydata, 1000)  # number of iterations

# tell console training is complete
print('Training complete')


# %% setup colour and labels

# TODO edit this so that appended zero array is correct
# load colour labels
target = np.genfromtxt(r'H:\MSc\project\data\yvette_02-09-20\High Wavenumbers for Dan.csv', delimiter=',', usecols=(0), dtype=str)
"""
t = np.zeros(len(target), dtype=int)
t[target == 'R'] = 0
t[target == 'G'] = 1
"""
t = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# assign colours and markers to each label
markers = ['o', 'o']  # add markers to data
colors = ['r', 'g']  # edit marker colours


# %% plot distance map (u-matrix) and overlay mapped sample

# initialise figure canvas with SOM
plt.figure()
plt.pcolor(som.distance_map().T, cmap='Blues')  # plot SOM distances in one matrix, transpose distances using .T, and set colourmap
plt.colorbar()  # add legend of normalised values

# calculate and plot BMU for sample
for cnt, xx in enumerate(nydata):
    bmu = som.winner(xx)  # calculate BMU
    plt.plot(bmu[0]+.5, bmu[1]+.5, markers[t[cnt]], markerfacecolor=colors[t[cnt]], markeredgecolor=colors[t[cnt]], markersize=6, markeredgewidth=2)  # place marker on winning position for sample xx
plt.axis([0, som._weights.shape[0], 0, som._weights.shape[1]])

plt.show()


# %% plot scatter plot of dots representing co-ordinates of winning neuron across map with random offset to avoid overlaps between pojnts within same cell

# generate scatter plot data
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
    plt.scatter(w_x[idx_t]+.5+(np.random.rand(np.sum(idx_t))-.5)*.8, w_y[idx_t]+.5+(np.random.rand(np.sum(idx_t))-.5)*.8, s=50, c=colors[c])

plt.show()
