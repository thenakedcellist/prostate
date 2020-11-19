import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def frob_norm(ydata):
    """normalise data - MiniSom method dividing each column by Frobenius norm"""
    nydata = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, ydata)
    return nydata


def sklearn_norm(ydata):
    """normalise data - sklearn method scaling to 0 mean and unit variance"""
    sydata = preprocessing.scale(ydata)
    return sydata


# yvette's data
yvette_xdata = np.genfromtxt('../../data/yvette_02_09_20/xwavehw.csv', delimiter=',')
yvette_ydata = np.genfromtxt('../../data/yvette_02_09_20/High Wavenumbers for Dan.csv', delimiter=',', usecols=np.arange(1, 1057))
yvette_nydata = frob_norm(yvette_ydata)
yvette_sydata = sklearn_norm(yvette_ydata.T).T

# banbury_data_extractor data
banbury_xdata = np.genfromtxt('../../data/banbury/_data_1-6/_wavenumbers.csv', delimiter=',', usecols=np.arange(1, 1016))
banbury_ydata = np.genfromtxt('../../data/banbury/_data_1-6/all_data.csv', delimiter=',', usecols=np.arange(1, 1016))
banbury_nydata = frob_norm(banbury_ydata)
banbury_sydata = sklearn_norm(banbury_ydata)


# yvette's frobenius normalised data
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
fig.suptitle('normalisation methods')
ax1.plot(yvette_xdata, yvette_nydata[0])
ax2.plot(yvette_xdata, yvette_sydata[0])
ax3.plot(banbury_xdata, banbury_nydata[0])
ax4.plot(banbury_xdata, banbury_sydata[0])
plt.show()

from sklearn import preprocessing

nala = np.array([[1, 2, 3, 4, 5],
                [1, 4, 4, 4, 4],
                [5, 4, 3, 3, 2]])

nico = preprocessing.scale(nala)

nico.mean(axis=0)
nico.std(axis=0)