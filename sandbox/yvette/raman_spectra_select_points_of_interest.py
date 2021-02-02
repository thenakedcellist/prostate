import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mysom.mysom import MySom
from pathlib import Path

x_path = Path('../../../data/yvette_20_09_02/xwavehw.csv')
y_path = Path('../../../data/yvette_20_11_18/shuffled_data_named.csv')
figpath = Path('../../thesis/yvette/Experiment 1/img_raman_spectra/')
datestr = '2020_12_08'

x_data = np.genfromtxt(x_path, delimiter=',')
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 1057))

label_list = ['PNT2', 'LNCaP']
marker_list = ['o', 'x']
colour_list = ['#FFA500', 'g']

somI = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=1)
somI.frobenius_norm_normalisation(y_data)
somI.make_som(10000)
somI.make_labels(y_path, label_list, marker_list, colour_list)
somI.plot_som_scatter(figpath, datestr, onlyshow=True)
somI.plot_density_function(figpath, datestr, onlyshow=True)
winmap = somI.som.win_map(somI.nydata, return_indices=True)
labmap = somI.som.labels_map(somI.nydata, somI.t)

# looking at density plot, there is a large concentration of points in (8, 8),
# and the central peak spans (7, 7) to (8, 8)

# generate list of lists of point of interest from cells within dense region
poi = [winmap[(7, 7)], winmap[(7, 8)], winmap[(8, 7)], winmap[(8, 8)]]
# flatten list of lists into list
flat_poi = [item for sublist in poi for item in sublist]
# generate list of corresponding label values for points of interest
labs_poi = [somI.t[i] for i in flat_poi]
# generate lists of poi indices by grouping
PNT2_poi = []
LNCaP_poi = []
for i, v in enumerate(labs_poi):
    if v == 0:
        PNT2_poi.append(flat_poi[i])
    elif v == 1:
        LNCaP_poi.append(flat_poi[i])
# generate outlier points of interest indices list
outlier_poi = winmap[(6, 0)]
# generate lists of all data indices by group
PNT2_all = []
LNCaP_all = []
for i, v in enumerate(somI.t):
    if v == 0:
        PNT2_all.append(i)
    elif v == 1:
        LNCaP_all.append(i)
# generate list of group indices excluding outlier points of interest
PNT2_no_outlier = [v for v in PNT2_all for o in outlier_poi if v != o]

nxdata = x_data[None, :]

# generate arrays containing data from points of interest and all data by group
PNT2_all_data = y_data[np.ix_(PNT2_all)]
LNCaP_all_data = y_data[np.ix_(LNCaP_all)]
PNT2_poi_data = y_data[np.ix_(PNT2_poi)]
LNCaP_poi_data = y_data[np.ix_(LNCaP_poi)]
outlier_poi_data = y_data[np.ix_(outlier_poi)]
PNT2_no_outlier_data = y_data[np.ix_(PNT2_no_outlier)]

# generate column means for data groups
PNT2_all_column_mean = PNT2_all_data.mean(axis=0)
LNCaP_all_column_mean = LNCaP_all_data.mean(axis=0)
PNT2_poi_column_mean = PNT2_poi_data.mean(axis=0)
LNCaP_poi_column_mean = LNCaP_poi_data.mean(axis=0)
PNT2_no_outlier_column_mean = PNT2_no_outlier_data.mean(axis=0)

# plot average spectra from dense region
fig1, ax1 = plt.subplots(1, 1)
fig1.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)  # set whitespace around figure edges and space between subplots
fig1.suptitle("Average Raman Spectra from Dense Region by Cell Line", fontsize=14)
ax1.plot(x_data, PNT2_poi_column_mean, color='#FFA500', label='PNT2')
ax1.plot(x_data, LNCaP_poi_column_mean, color='g', label='LNCaP')
ax1.set(xlabel='Wavenumber', ylabel='Intensity')
legend_elements1 = [(Line2D([], [], linestyle='-', linewidth=1, color='#FFA500', label='PNT2')),
                   (Line2D([], [], linestyle='-', linewidth=1, color='g', label='LNCaP'))]
ax1.legend(handles=legend_elements1, loc='upper right', bbox_to_anchor=(0.99, 0.97), borderaxespad=0, ncol=1, fontsize=10)
fig1.show()
fig1.savefig(figpath / 'eps' / 'som8_Average_Raman_Spectra_from_Dense_Region_by_Cell_Line.eps', format='eps')
fig1.savefig(figpath / 'png' / 'som8_Average_Raman_Spectra_from_Dense_Region_by_Cell_Line.png', format='png')

# plot all spectra from dense region
fig2, ax2 = plt.subplots(1, 1)
fig2.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)  # set whitespace around figure edges and space between subplots
fig2.suptitle("All Raman Spectra from Dense Region by Cell Line", fontsize=14)
for i in range(len(PNT2_poi_data)):
    ax2.plot(x_data, PNT2_poi_data[i], color='#FFA500', label='PNT2')
for i in range(len(LNCaP_poi_data)):
    ax2.plot(x_data, LNCaP_poi_data[i], color='g', label='LNCaP')
ax2.set(xlabel='Wavenumber', ylabel='Intensity')
legend_elements2 = [(Line2D([], [], linestyle='-', linewidth=1, color='#FFA500', label='PNT2')),
                   (Line2D([], [], linestyle='-', linewidth=1, color='g', label='LNCaP'))]
ax2.legend(handles=legend_elements2, loc='upper right', bbox_to_anchor=(0.99, 0.97), borderaxespad=0, ncol=1, fontsize=10)
fig2.show()
fig2.savefig(figpath / 'eps' / 'som8_All_Raman_Spectra_from_Dense_Region_by_Cell_Line.eps', format='eps')
fig2.savefig(figpath / 'png' / 'som8_All_Raman_Spectra_from_Dense_Region_by_Cell_Line.png', format='png')

# plot outlier spectrum against all spectra from its group
fig3, ax3 = plt.subplots(1, 1)
fig3.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)  # set whitespace around figure edges and space between subplots
fig3.suptitle("All Raman Spectra from PNT2 and Outlier Spectrum", fontsize=14)
for i in range(len(PNT2_all_data)):
    ax3.plot(x_data,  PNT2_all_data[i], color='#FFA500', label='PNT2')
for i in range(len(outlier_poi_data)):
    ax3.plot(x_data, outlier_poi_data[i], color='#00BFFF', label='Outlier')
ax3.set(xlabel='Wavenumber', ylabel='Intensity')
legend_elements3 = [(Line2D([], [], linestyle='-', linewidth=1, color='#FFA500', label='PNT2')),
                   (Line2D([], [], linestyle='-', linewidth=1, color='#00BFFF', label='Outlier'))]
ax3.legend(handles=legend_elements3, loc='upper right', bbox_to_anchor=(0.99, 0.97), borderaxespad=0, ncol=1, fontsize=10)
fig3.show()
fig3.savefig(figpath / 'eps' / 'som8_All_Raman_Spectra_from_PNT2_and_Outlier_Spectrum.eps', format='eps')
fig3.savefig(figpath / 'png' / 'som8_All_Raman_Spectra_from_PNT2_and_Outlier_Spectrum.png', format='png')

# plot outlier spectrum against average of all other spectra from its group
fig4, ax4 = plt.subplots(1, 1)
fig4.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)  # set whitespace around figure edges and space between subplots
fig4.suptitle("Average PNT2 Spectrum and Outlier Spectrum", fontsize=14)
ax4.plot(x_data, PNT2_no_outlier_column_mean, color='#FFA500', label='PNT2')
for i in range(len(outlier_poi_data)):
    ax4.plot(x_data, outlier_poi_data[i], color='#00BFFF', label='Outlier')
ax4.set(xlabel='Wavenumber', ylabel='Intensity')
legend_elements4 = [(Line2D([], [], linestyle='-', linewidth=1, color='#FFA500', label='PNT2')),
                   (Line2D([], [], linestyle='-', linewidth=1, color='#00BFFF', label='Outlier'))]
ax4.legend(handles=legend_elements4, loc='upper right', bbox_to_anchor=(0.99, 0.97), borderaxespad=0, ncol=1, fontsize=10)
fig4.show()
fig4.savefig(figpath / 'eps' / 'som8_Average_PNT2_Spectrum_and_Outlier_Spectrum.eps', format='eps')
fig4.savefig(figpath / 'png' / 'som8_Average_PNT2_Spectrum_and_Outlier_Spectrum.png', format='png')

# plot all spectra by group with outlier spectrum separately
fig5, ax5 = plt.subplots(1, 1)
fig5.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)  # set whitespace around figure edges and space between subplots
fig5.suptitle("All Raman Spectra by Cell Line and Outlying Spectrum", fontsize=14)
for i in range(len(PNT2_all_data)):
    ax5.plot(x_data, PNT2_all_data[i], color='#FFA500', label='PNT2')
for i in range(len(LNCaP_all_data)):
    ax5.plot(x_data, LNCaP_all_data[i], color='g', label='LNCaP')
for i in range(len(outlier_poi_data)):
    ax5.plot(x_data, outlier_poi_data[i], color='#00BFFF', label='Outlier')
ax5.set(xlabel='Wavenumber', ylabel='Intensity')
legend_elements5 = [(Line2D([], [], linestyle='-', linewidth=1, color='#FFA500', label='PNT2')),
                    (Line2D([], [], linestyle='-', linewidth=1, color='g', label='LNCaP')),
                    (Line2D([], [], linestyle='-', linewidth=1, color='#00BFFF', label='Outlier'))]
ax5.legend(handles=legend_elements5, loc='upper right', bbox_to_anchor=(0.99, 0.97), borderaxespad=0, ncol=1, fontsize=10)
fig5.show()
fig5.savefig(figpath / 'eps' / 'som8_All_Raman_Spectra_by_Cell_Line_and_Outlying_Spectrum.eps', format='eps')
fig5.savefig(figpath / 'png' / 'som8_All_Raman_Spectra_by_Cell_Line_and_Outlying_Spectrum.png', format='png')
