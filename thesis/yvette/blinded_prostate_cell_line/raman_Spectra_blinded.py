import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mysom.mysom import MySom
from pathlib import Path

def all_data_points_by_group(som):
    """generate lists of all data indices by group"""
    PNT2_all = []
    LNCaP_all = []
    for i, v in enumerate(som.t):
        if v == 1:
            PNT2_all.append(i)
        elif v == 2:
            LNCaP_all.append(i)
    return PNT2_all, LNCaP_all

def points_of_interest_by_group(som, poi):
    """generate lists of data indices of points of interest by group"""
    # generate list of corresponding label values for points of interest
    labs_poi = [som.t[i] for i in poi]
    # generate lists of point of interest indices by group
    PNT2_poi = []
    LNCaP_poi = []
    for i, v in enumerate(labs_poi):
        if v == 1:
            PNT2_poi.append(poi[i])
        elif v == 2:
            LNCaP_poi.append(poi[i])
    return PNT2_poi, LNCaP_poi

def remove_poi_from_list(list_a, list_b):
    """takes two lists A and B, and removes the values of list B from list A"""
    concat_list = [v for v in list_a if v not in list_b]
    return concat_list

def make_poi_array(input_list):
    """generate arrays of input data with indices from input list"""
    output_arr = y_data[np.ix_(input_list)]
    return output_arr

def make_array_column_mean(input_arr):
    """calculate column mean of input array"""
    output_arr = input_arr.mean(axis=0)[None, :]
    return output_arr

def make_array_column_standard_deviation(input_arr):
    """calculate column mean of input array"""
    output_arr = input_arr.std(axis=0)[None, :]
    return output_arr

def make_array_column_variance(input_arr):
    """calculate column mean of input array"""
    output_arr = input_arr.var(axis=0)[None, :]
    return output_arr

def make_array_column_maximum(input_arr):
    """calculate column mean of input array"""
    output_arr = np.amax(input_arr, axis=0)[None, :]
    return output_arr

def make_array_column_minimum(input_arr):
    """calculate column mean of input array"""
    output_arr = np.amin(input_arr, axis=0)[None, :]
    return output_arr

def save_figure(fig, title_string):
    """save fig as eps and png with title_string in figpath"""
    fig.savefig(figpath / 'eps' / f'{title_string}.eps', format='eps')
    fig.savefig(figpath / 'png' / f'{title_string}.png', format='png')
    return


x_path = Path('../../../../data/yvette_20_09_02/xwavehw.csv')
y_path = Path('../../../../data/yvette_20_11_18/shuffled_data_unnamed.csv')
figpath = Path('img_raman_spectra/')
datestr = '2021_02_28'
x_data = np.genfromtxt(x_path, delimiter=',')[None, :]
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(0, 1056))

label_list = ['(112)', 'Blinded Data']
marker_list = ['d', 'o']
colour_list = ['#00BFFF', '#FFA500']

# putative outlier signal is index 46
removal_list = [46]

sample_outlier_removed = np.delete(y_data, 46, axis=0)

# plot of outlier spectrum and rest of dataset
fig1, ax1 = plt.subplots(1, 1)
fig1.subplots_adjust(left=0.125, right=0.9, top=0.8, bottom=0.1, wspace=0.2, hspace=0.2)
fig1.suptitle("All Spectra with Outlier Marked", fontsize=14)
for i in range(len(sample_outlier_removed)):
    ax1.plot(x_data[0, :], sample_outlier_removed[i], color='#FFA500', linestyle='--', label='blinded')
ax1.plot(x_data[0, :], y_data[46], color='#00FFFF', linestyle='-', label='outlier')
ax1.set(xlabel='Wavenumber (cm$^{-1}$)', ylabel='Intensity')
legend_elements1 = [(Line2D([], [], linestyle='-', linewidth=1, color='#00FFFF', label='Outlier')),
                    (Line2D([], [], linestyle='--', linewidth=1, color='#FFA500', label='Blinded Data'))]
ax1.legend(handles=legend_elements1, loc='lower left', bbox_to_anchor=(0.0, 1.04), borderaxespad=0, ncol=2, fontsize=10)
fig1.show()

# Average spectra with outlier removed
sample_mean = make_array_column_mean(y_data)
sample_outlier_removed_mean = make_array_column_mean(sample_outlier_removed)

fig2, ax2 = plt.subplots(1, 1)
fig2.subplots_adjust(left=0.125, right=0.9, top=0.8, bottom=0.1, wspace=0.2, hspace=0.2)
fig2.suptitle("Sample Mean with Outlier Removed", fontsize=14)
ax2.plot(x_data[0, :], sample_mean[0, :], color='#00FFFF', linestyle='-',  label='sample_mean')
ax2.plot(x_data[0, :], sample_outlier_removed_mean[0, :], color='#000000', linestyle='--',  label='sample_mean_outlier_removed')
ax2.set(xlabel='Wavenumber (cm$^{-1}$)', ylabel='Intensity')
legend_elements2 = [(Line2D([], [], linestyle='-', linewidth=1, color='#00FFFF', label='Sample Mean')),
                    (Line2D([], [], linestyle='--', linewidth=1, color='#000000', label='Sample without Outlier Mean'))]
ax2.legend(handles=legend_elements2, loc='lower left', bbox_to_anchor=(0.0, 1.04), borderaxespad=0, ncol=2, fontsize=10)
fig2.show()
