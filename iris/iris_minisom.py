# %% imports and setup

from numpy import genfromtxt, array, linalg, zeros, apply_along_axis
from minisom import MiniSom
from pylab import plot, axis, show, pcolor, colorbar, bone

# %% reading adn normalisation of dataset

# read iris database in csv format
data = genfromtxt('iris.csv', delimiter=',', usecols=(0, 1, 2, 3))

# normalisation of dataset to unity
data = apply_along_axis(lambda x: x/linalg.norm(x), 1, data)


# %% SOM training

# intialise SOM with random weights
som = MiniSom(40, 40, 4, sigma=1.0, learning_rate=0.1)
som.random_weights_init(data)

# train SOM and tell console training is in progress
print("Training...")
som.train_random(data, 250)  # training with 250 iterations

# tell console trianing is complete
print("Training complete")


# %% visualise output of training

bone()  # set colormap to bone
pcolor(som.distance_map().T, cmap='Reds')  # distance map set to background
colorbar()

# load iris labels
target = genfromtxt('iris.csv', delimiter=',', usecols=(4), dtype=str)
t = zeros(len(target), dtype=int)
t[target == 'sertosa'] = 0
t[target == 'versicolor'] = 1
t[target == 'virginica'] = 2

# assign different colours and markers to each label
markers = ['o', 's', 'D']
colors = ['r', 'g', 'b']

# calculate and plot BMU for sample
for cnt, xx in enumerate(data):
    bmu = som.winner(xx)  # calculate BMU
    plot(bmu[0]+.5, bmu[1]+.5, markers[t[cnt]], markerfacecolor='None', markeredgecolor=colors[t[cnt]], markersize=12, markeredgewidth=2)  # place marker on winning position for sample xx
axis([0, som._weights.shape[0], 0, som._weights.shape[1]])  # generate plot
show()  # display figure
