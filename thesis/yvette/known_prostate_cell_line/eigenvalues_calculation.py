import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path

x_path = Path('../../../../data/yvette_20_09_02/xwavehw.csv')
y_path = Path('../../../../data/yvette_20_09_02/High Wavenumbers for Dan.csv')
figpath = Path('img_som_plots/')
datestr = '2021_03_18'

x_data = np.genfromtxt(x_path, delimiter=',')[None, :]
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 1057))

label_list = ['PNT2-C2', 'LNCaP']
marker_list = ['o', 'x']
colour_list = ['#FFA500', '#FF00FF']

# plot spectra
fig1, ax1 = plt.subplots(1, 1)
fig1.subplots_adjust(left=0.125, right=0.9, top=0.8, bottom=0.1, wspace=0.2, hspace=0.2)
fig1.suptitle("All Spectra with Outlier Removed", fontsize=14)
for spectrum in range(len(y_data)):
    ax1.plot(x_data[0, :], y_data[spectrum], color='#00FFFF', linestyle='-', label='outlier')
ax1.set(xlabel='Wavenumber (cm$^{-1}$)', ylabel='Intensity')
legend_elements1 = [Line2D([], [], linestyle='-', linewidth=1, color='#00FFFF', label='y_data')]
ax1.legend(handles=legend_elements1, loc='lower left', bbox_to_anchor=(0.0, 1.04), borderaxespad=0, ncol=2, fontsize=10)
fig1.show()

# generate autocorrelation matrix as pd dataframe
df = pd.DataFrame(y_data)
corrMatrix = df.corr()
print(corrMatrix)

# convert autocorrelation matrix pd dataframe to np array
corrMatrixArr = np.array(corrMatrix)

# genereate eigenvalues adn eigenvectors of autocorrrelation matrix
eigvals, eigvects = np.linalg.eig(corrMatrixArr)
eigvals = eigvals.real  #matrix is sqaure so eigenvalues are real

sorted_eigvals = sorted(eigvals, reverse=True)
a = sorted_eigvals[0]
b = sorted_eigvals[1]
print(f"optimum map dimension ratio is {a} to {b} = {a / b}")
