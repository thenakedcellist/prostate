# %% setup

from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt


# %% read data

# read data in csv format
data = np.genfromtxt(r'H:\MSc\project\practice_datasets\colours_dataset_small\colours_edit_hex.csv', delimiter=',', skip_header=1, usecols=(0, 1, 2))


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
plt.figure(figsize=((16/2.54), (12/2.52)))
plt.pcolor(som.distance_map().T, cmap='Greys')  # plot distances in one matrix, transpose distances using .T, and set colourmap
plt.colorbar()  # add legend of normalised values

# load colour labels
colour = np.genfromtxt(r'H:\MSc\project\practice_datasets\colours_dataset_small\colours_edit_hex.csv', delimiter=',', skip_header=1, usecols=(3), dtype=str)


# %% visualise BMUs

# calculate and plot BMU for sample
for cnt, xx in enumerate(ndata):
    bmu = som.winner(xx)  # calculate BMU
    plt.plot(bmu[0]+.5, bmu[1]+.5, marker = 'o', markerfacecolor= colour[cnt], markeredgecolor= 'k', markersize=4, markeredgewidth=1)  # place marker on winning position for sample xx
plt.axis([0, som._weights.shape[0], 0, som._weights.shape[1]])  # generate plot
plt.show()  # display plot



# %% TODO
"""

add hex code to output plot points - unsure how to acheive this
Do we need a random seed defined for learning - almost definitely
Add pickling to save and load SOM

"""