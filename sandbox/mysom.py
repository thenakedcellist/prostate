'''

don't forget to look at time complexity - scalene, scalene, scalene, scaleeeeeeeene

'''

import numpy as np
from sklearn import preprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import minisom

xdata = np.genfromtxt('../data/banbury/_data_1-6/_wavenumbers.csv', delimiter=',', usecols=np.arange(1, 1016))
ydata = np.genfromtxt('../data/banbury/_data_1-6/all_data.csv', delimiter=',', usecols=np.arange(1, 1016))

# 5 * sqrt(2640) = 256.90 so x = 16 y = 16 input_len = ydata.shape[1]

def frobenius_norm(ydata):
    """normalise data by dividing each column by the Frobenius norm"""
    nydata = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, ydata)
    return nydata


def make_som(x, y, input_len):
    """implement MiniSom via mysom and allow manipulation of key parameters"""
    x = x
    y = y
    input_len = input_len
    sigma = 1.0
    learning_rate = 0.5
    decay_function = 'asymptotic_decay'
    neighborhood_function = 'gaussian'
    activation_distance = 'euclidean'
    topology = 'rectangular'
    random_seed = 1
    train_iter = 1000
    som = minisom.MiniSom(x, y, input_len, sigma, learning_rate,
                 decay_function, neighborhood_function, topology,
                 activation_distance, random_seed)
    som.random_weights_init(nydata)
    som.train_random(nydata, train_iter)
    return som


def make_labels(label_list, colour_list):
    """generate labels for som"""
    target = np.gen