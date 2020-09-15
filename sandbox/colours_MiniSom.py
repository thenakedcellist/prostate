from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt


# %% read from dataset

# read dataset from file
data = np.genfromtxt('rgb_standard_colours_labels.csv', delimiter=',', skip_header=1, usecols=(0, 1, 2))

# normalisation to unity of each pattern in data
data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)


# %% initialisation and training of SOM

# SOM initialisation
som = MiniSom(x=15, y=15, input_len=3, sigma=1.0, learning_rate=0.1)
som.random_weights_init(data)
starting_weights = som.get_weights().copy()  # record starting weights

# SOM training
print('Training...')
som.train_random(data, 100)  # 100 iterations
print('Ready to go!')


# %% vector quantization

# vector quantization
qnt = som.quantization(data)


# %% visualise SOM

# create empty matrix with original deimensionality
target = np.genfromtxt('rgb_standard_colours_labels.csv', delimiter=',', skip_header=1, usecols=(0, 1, 2), dtype=int)
t = np.zeros(data.shape, dtype=int)  # empty matrix

# generate SOM
for i, q in enumerate(qnt):
    t[np.unravel_index(i, shape=(data[0], data[1]))] = q

# code based on https://heartbeat.fritz.ai/introduction-to-self-organizing-maps-soms-98e88b568f5d


"""
# display plots
plt.figure(figsize=(12, 6))
"""
