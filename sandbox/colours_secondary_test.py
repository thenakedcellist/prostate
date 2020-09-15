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
som = MiniSom(7, 7, 3, sigma=1.0, learning_rate=0.1, random_seed=1)
som.random_weights_init(ndata)

# train SOM and tell console that training is in progress
print('Training...')
som.train_random(ndata, 1000)  # number of iterations

# tell console training is complete
print('Training complete')


# %% create output grid and visualise output of training

# initialise figure canvas
plt.figure(figsize=((24/2.54), (18/2.52)))
plt.pcolor(som.distance_map().T, cmap='Greys')  # plot distances in one matrix, transpose distances using .T, and set colourmap
plt.colorbar()  # add legend of normalised values

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
plt.markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o']  # add markers to data
plt.colors = ['r', 'g', 'b', 'k', 'k', 'k', 'k']  # edit marker colours


# %% visualise BMUs

# calculate and plot BMU for sample
for cnt, xx in enumerate(ndata):
    bmu = som.winner(xx)  # calculate BMU
    plt.plot(bmu[0]+.5, bmu[1]+.5, plt.markers[t[cnt]], markerfacecolor=plt.colors[t[cnt]], markeredgecolor=plt.colors[t[cnt]], markersize=6, markeredgewidth=2)  # place marker on winning position for sample xx
plt.axis([0, som._weights.shape[0], 0, som._weights.shape[1]])  # generate plot
plt.show()  # display plot


# %% plot secondary colours

# initialise figure canvas
plt.figure(figsize=((24/2.54), (18/2.52)))
plt.pcolor(som.distance_map().T, cmap='Greys')  # plot distances in one matrix, transpose distances using .T, and set colourmap
plt.colorbar()  # add legend of normalised values

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
plt.markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o']  # add markers to data
plt.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # edit marker colours

# calculate and plot BMU for sample
for cnt, xx in enumerate(ndata):
    bmu = som.winner(xx)  # calculate BMU
    plt.plot(bmu[0]+.5, bmu[1]+.5, plt.markers[t[cnt]], markerfacecolor=plt.colors[t[cnt]], markeredgecolor=plt.colors[t[cnt]], markersize=6, markeredgewidth=2)  # place marker on winning position for sample xx
plt.axis([0, som._weights.shape[0], 0, som._weights.shape[1]])  # generate plot
plt.show()  # display plot