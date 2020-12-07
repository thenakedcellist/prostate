import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

x_path = Path('../../../../data/yvette_20_09_02/xwavehw.csv')
y_path = Path('../../../../data/yvette_20_09_02/High Wavenumbers for Dan.csv')
figpath = Path('img/')

x_data = np.genfromtxt(x_path, delimiter=',')
y_data_PNT2 = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 1057), skip_footer=15)
y_data_LNCaP = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 1057), skip_header=15)

PNT2_column_mean = y_data_PNT2.mean(axis=0)
LNCaP_column_mean = y_data_LNCaP.mean(axis=0)

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
fig.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)  # set whitespace around figure edges and space between subplots
fig.suptitle("Average Raman Spectra from Prostate Cell Lines", fontsize=14)
ax.plot(x_data, PNT2_column_mean, color='#00BFFF', label='Normal Prostate')
ax.plot(x_data, LNCaP_column_mean, color='#FFA500', label='Prostate Cancer')
ax.set(xlabel='Wavenumber', ylabel='Intensity')
legend_elements = [(Line2D([], [], linestyle='-', linewidth=1, color='#00BFFF', label='Normal Prostate')),
                   (Line2D([], [], linestyle='-', linewidth=1, color='#FFA500', label='Prostate Cancer'))]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.97), borderaxespad=0,
          ncol=1, fontsize=10)
ax.annotate()
fig.show()
'''
fig.savefig(figpath / 'eps' / 'average_raman_spectra_cell_lines.eps')
fig.savefig(figpath / 'png' / 'average_raman_spectra_cell_lines.eps')

fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.4))
fig.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)  # set whitespace around figure edges and space between subplots
fig.suptitle("Average Raman Spectra from Prostate Cell Lines", fontsize=14)
ax.plot(x_data, PNT2_column_mean, color='#00BFFF', label='Normal Prostate')
ax.plot(x_data, LNCaP_column_mean, color='#FFA500', label='Prostate Cancer')
ax.set(xlabel='Wavenumber', ylabel='Intensity')
legend_elements = [(Line2D([], [], linestyle='-', linewidth=1, color='#00BFFF', label='Normal Prostate')),
                   (Line2D([], [], linestyle='-', linewidth=1, color='#FFA500', label='Prostate Cancer'))]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.97), borderaxespad=0,
          ncol=1, fontsize=10)
fig.show()
fig.savefig(figpath / 'eps' / 'average_raman_spectra_cell_lines_beamer.eps')
fig.savefig(figpath / 'png' / 'average_raman_spectra_cell_lines_beamer.eps')
'''