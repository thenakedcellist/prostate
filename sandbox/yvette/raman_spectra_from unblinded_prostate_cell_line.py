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
        if v == 0:
            PNT2_all.append(i)
        elif v == 1:
            LNCaP_all.append(i)
    return PNT2_all, LNCaP_all

def points_of_interest_by_group(som, poi):
    """generate lists of data indices of points of interest by group"""
    # flatten list of points of interest
    flat_poi = [item for sublist in poi for item in sublist]
    # generate list of corresponding label values for points of interest
    labs_poi = [som.t[i] for i in flat_poi]
    # generate lists of point of interest indices by group
    PNT2_poi = []
    LNCaP_poi = []
    for i, v in enumerate(labs_poi):
        if v == 0:
            PNT2_poi.append(flat_poi[i])
        elif v == 1:
            LNCaP_poi.append(flat_poi[i])
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

def plot_spectra_from_poi(cluster_PNT2_input_arr, cluster_LNCaP_input_arr, title_string, onlyshow=False):
    """plot average spectra for cluster by group"""
    fig, ax = plt.subplots(1, 1)
    # set whitespace around figure edges and space between subplots
    fig.subplots_adjust(left=0.125, right=0.9, top=0.8, bottom=0.1, wspace=0.2, hspace=0.2)
    fig.suptitle(title_string, fontsize=14)
    for i in range(len(cluster_PNT2_input_arr)):
        ax.plot(x_data[0, :] , cluster_PNT2_input_arr[i], color='#FFA500', label='PNT2')
    for i in range(len(cluster_LNCaP_input_arr)):
        ax.plot(x_data[0, :] , cluster_LNCaP_input_arr[i], color='g', label='LNCaP')
    ax.set(xlabel='wavenumber', ylabel='intensity')
    legend_elements = [(Line2D([], [], linestyle='-', linewidth=1, color='#FFA500', label='PNT2')),
                       (Line2D([], [], linestyle='-', linewidth=1, color='g', label='LNCaP'))]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.0, 1.04), borderaxespad=0, ncol=2,
              fontsize=10)
    fig.show()
    if onlyshow:
        pass
    else:
        fig.savefig(figpath / 'eps' / f'{title_string}.eps', format='eps')
        fig.savefig(figpath / 'png' / f'{title_string}.png', format='png')


x_path = Path('../../../data/yvette_20_09_02/xwavehw.csv')
y_path = Path('../../../data/yvette_20_11_18/shuffled_data_named.csv')
figpath = Path('../../thesis/yvette/blinded_prostate_cell_line/img_raman_spectra/')
datestr = '2021_02_02'
x_data = np.genfromtxt(x_path, delimiter=',')[None, :]
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 1057))

label_list = ['PNT2', 'LNCaP']
marker_list = ['o', 'x']
colour_list = ['#FFA500', 'g']

# putative noisy signal is index 46
removal_list = [46]

# full som
som = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=1)
som.som_setup(x_data, y_data, y_path, label_list, marker_list, colour_list)
som.frobenius_norm_normalisation()
som.train_som(10000)
som.plot_som_scatter(figpath, datestr, onlyshow=True)
som.plot_density_function(figpath, datestr, onlyshow=True)
winmap = som.som.win_map(som.nydata, return_indices=True)
labmap = som.som.labels_map(som.nydata, som.t)

# som with outlier removed
som_outlier_removed = MySom(x=9, y=9, input_len=y_data.shape[1], sigma=3.0, learning_rate=1.0, topology='rectangular', random_seed=1)
som_outlier_removed.som_setup(x_data, y_data, y_path, label_list, marker_list, colour_list)
som_outlier_removed.remove_observations_from_input_data(removal_list)
som_outlier_removed.frobenius_norm_normalisation()
som_outlier_removed.train_som(10000)
som_outlier_removed.plot_som_scatter(figpath, datestr, onlyshow=True)
som_outlier_removed.plot_density_function(figpath, datestr, onlyshow=True)
winmap = som_outlier_removed.som.win_map(som_outlier_removed.nydata, return_indices=True)
labmap = som_outlier_removed.som.labels_map(som_outlier_removed.nydata, som_outlier_removed.t)

# clusters
poi_a = [winmap[(0, 6)], winmap[(1, 6)], winmap[(0, 7)], winmap[(1, 7)], winmap[(0, 8)], winmap[(1, 8)]]
poi_b = [winmap[(7, 7)], winmap[(8, 7)], winmap[(7, 8)], winmap[(8, 8)]]
poi_c = [winmap[(3, 1)], winmap[(4, 1)], winmap[(3, 2)], winmap[(4, 2)]]
poi_outlier = [winmap[(6, 0)]]

# runcode

# all data points
all_PNT2_list, all_LNCaP_list = all_data_points_by_group(som)
all_PNT2_arr = make_poi_array(all_PNT2_list)
all_PNT2_mean = make_array_column_mean(all_PNT2_arr)
all_LNCaP_arr = make_poi_array(all_LNCaP_list)
all_LNCaP_mean = make_array_column_mean(all_LNCaP_arr)

# cluster A - top left
a_PNT2_list, a_LNCaP_list = points_of_interest_by_group(som, poi_a)
a_PNT2_arr = make_poi_array(a_PNT2_list)
a_PNT2_mean = make_array_column_mean(a_PNT2_arr)
a_LNCaP_arr = make_poi_array(a_LNCaP_list)
a_LNCaP_mean = make_array_column_mean(a_LNCaP_arr)

# cluster B - top right
b_PNT2_list, b_LNCaP_list = points_of_interest_by_group(som, poi_b)
b_PNT2_arr = make_poi_array(b_PNT2_list)
b_PNT2_mean = make_array_column_mean(b_PNT2_arr)
b_LNCaP_arr = make_poi_array(b_LNCaP_list)
b_LNCaP_mean = make_array_column_mean(b_LNCaP_arr)

# cluster C - bottom centre
c_PNT2_list, c_LNCaP_list = points_of_interest_by_group(som, poi_c)
c_PNT2_arr = make_poi_array(c_PNT2_list)
c_PNT2_mean = make_array_column_mean(c_PNT2_arr)
c_LNCaP_arr = make_poi_array(c_LNCaP_list)
c_LNCaP_mean = make_array_column_mean(c_LNCaP_arr)

# all data points without outlier
all_PNT2_no_outlier_list, all_LNCaP_no_outlier_list = all_data_points_by_group(som_outlier_removed)
all_PNT2_no_outlier_arr = make_poi_array(all_PNT2_no_outlier_list)
all_PNT2_no_outlier_mean = make_array_column_mean(all_PNT2_no_outlier_arr)
all_LNCaP_no_outlier_arr = make_poi_array(all_LNCaP_no_outlier_list)
all_LNCaP_no_outlier_mean = make_array_column_mean(all_LNCaP_no_outlier_arr)

# plots
all_spectra = plot_spectra_from_poi(all_PNT2_arr, all_LNCaP_arr, "All Spectra by Group", onlyshow=True)
all_spectra_mean = plot_spectra_from_poi(all_PNT2_mean, all_LNCaP_mean, "Mean Spectrum by Group", onlyshow=True)
cluster_a = plot_spectra_from_poi(a_PNT2_arr, a_LNCaP_arr, "Cluster A Spectra by Group", onlyshow=True)
cluster_a_mean = plot_spectra_from_poi(a_PNT2_mean, a_LNCaP_mean, "Cluster A Mean Spectrum by Group", onlyshow=True)
cluster_b = plot_spectra_from_poi(b_PNT2_arr, b_LNCaP_arr, "Cluster B Spectra by Group", onlyshow=True)
cluster_b_mean = plot_spectra_from_poi(b_PNT2_mean, b_LNCaP_mean, "Cluster B Mean Spectrum by Group", onlyshow=True)
cluster_c = plot_spectra_from_poi(c_PNT2_arr, c_LNCaP_arr, "Cluster C Spectra by Group", onlyshow=True)
cluster_c_mean = plot_spectra_from_poi(c_PNT2_mean, c_LNCaP_mean, "Cluster C Mean Spectrum by Group", onlyshow=True)


all_spectra_no_outlier = plot_spectra_from_poi(all_PNT2_no_outlier_arr, all_LNCaP_no_outlier_arr, "", onlyshow=True)
all_spectra_no_outlier_mean = plot_spectra_from_poi(all_PNT2_no_outlier_mean, all_LNCaP_no_outlier_mean, "", onlyshow=True)
