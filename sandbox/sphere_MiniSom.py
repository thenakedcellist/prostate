from numpy import genfromtxt, array, linalg, zeros, apply_along_axis
from minisom import MiniSom
import matplotlib.pyplot as plt

# %% read from dataset

# read dataset from file
data = genfromtxt('sphere-rad1.csv', delimiter=',', skip_header=1, usecols=(0, 1, 2))

# normalisation to unity of each pattern in data
data = apply_along_axis(lambda x: x/linalg.norm(x), 1, data)


# %% initialisation and training of SOM

# SOM initialisation
som = MiniSom(20, 20, 3, sigma=1.0, learning_rate=0.5)
som.random_weights_init(data)

# SOM training
print('Training...')
som.train_random(data, 100)  # 100 iterations
print('Ready to go!')


# %% visualise SOM

# set distance map as background
plt.plot()
plt.pcolor(som.distance_map().T)
plt.colorbar()


