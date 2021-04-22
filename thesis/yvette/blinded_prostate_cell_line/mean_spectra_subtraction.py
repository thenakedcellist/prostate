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

def make_poi_array(som, input_list):
    """generate arrays of input data with indices from input list"""
    output_arr = som.ydata[np.ix_(input_list)]
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
y_path = Path('../../../../data/yvette_20_11_18/shuffled_data_named.csv')
figpath = Path('img_raman_spectra/')
datestr = 'som_unblinded_clusters'
x_data = np.genfromtxt(x_path, delimiter=',')[None, :]
y_data = np.genfromtxt(y_path, delimiter=',', usecols=np.arange(1, 1057))

label_list = ['PNT2', 'LNCaP']
marker_list = ['o', 'x']
colour_list = ['#FFA500', '#FF00FF']

# putative outlier signal is index 46
removal_list = [46]

sample_outlier_removed = np.delete(y_data, 46, axis=0)

# full som
som = MySom(x=23, y=6, input_len=y_data.shape[1], sigma=3.0, learning_rate=0.75, topology='rectangular', random_seed=1)
som.som_setup(x_data, y_data, y_path, label_list, marker_list, colour_list)
som.remove_observations_from_input_data(removal_list)
som.frobenius_norm_normalisation()
som.train_som(100000)
som.plot_som_scatter(figpath, datestr, onlyshow=True)
som.plot_density_function(figpath, datestr, onlyshow=True)
winmap = som.som.win_map(som.nydata, return_indices=True)
labmap = som.som.labels_map(som.nydata, som.t)

# poi lists
poi_a = [winmap[(2, 4)], winmap[(3, 4)], winmap[(2, 5)], winmap[(3, 5)]]
poi_b = [winmap[(11, 5)], winmap[(12, 5)], winmap[(13, 5)], winmap[(14, 5)], winmap[(15, 5)], winmap[(16, 5)],
         winmap[(17, 5)], winmap[(18, 5)], winmap[(19, 5)], winmap[(20, 5)], winmap[(21, 5)]]
poi_c = [winmap[(7, 0)], winmap[(8, 0)], winmap[(9, 0)], winmap[(10, 0)], winmap[(11, 0)], winmap[(12, 0)]]
# flat list
cluster_a = sorted([item for sublist in poi_a for item in sublist])
cluster_b = sorted([item for sublist in poi_b for item in sublist])
cluster_c = sorted([item for sublist in poi_c for item in sublist])

# cluster arrays and array means
cluster_a_arr = make_poi_array(som, cluster_a)
cluster_a_arr_mean = make_array_column_mean(cluster_a_arr)
cluster_b_arr = make_poi_array(som, cluster_b)
cluster_b_arr_mean = make_array_column_mean(cluster_b_arr)
cluster_c_arr = make_poi_array(som, cluster_c)
cluster_c_arr_mean = make_array_column_mean(cluster_c_arr)

# spectral subtraction of clusters a and b from cluster c
cluster_a_subtraction = cluster_c_arr_mean - cluster_a_arr_mean
cluster_b_subtraction = cluster_c_arr_mean - cluster_b_arr_mean

# plot of cluster a spectral subtraction
fig1, ax1 = plt.subplots(1, 1)
fig1.subplots_adjust(left=0.15, right=0.95, top=0.8, bottom=0.1, wspace=0.2, hspace=0.2)
fig1.suptitle("Cluster A Mean Subtraction", fontsize=14)
for i in range(len(cluster_a_subtraction)):
    ax1.plot(x_data[0, :], cluster_a_subtraction[i], color='#FF5380', linestyle='-', label='cluster_a_subtraction')
ax1.set(xlabel='Wavenumber (cm$^{-1}$)', ylabel='Intensity Difference')
legend_elements1 = [(Line2D([], [], linestyle='-', linewidth=1, color='#FF5380',
                            label='Cluster A Mean Subtracted from Cluster C Mean'))]
ax1.legend(handles=legend_elements1, loc='lower left', bbox_to_anchor=(0.0, 1.04), borderaxespad=0, ncol=2, fontsize=10)
fig1.show()

# plot of cluster b spectral subtraction
fig2, ax2 = plt.subplots(1, 1)
fig2.subplots_adjust(left=0.15, right=0.95, top=0.8, bottom=0.1, wspace=0.2, hspace=0.2)
fig2.suptitle("Cluster B Mean Subtraction", fontsize=14)
for i in range(len(cluster_b_subtraction)):
    ax2.plot(x_data[0, :], cluster_b_subtraction[i], color='#FF5380', linestyle='-', label='cluster_b_subtraction')
ax2.set(xlabel='Wavenumber (cm$^{-1}$)', ylabel='Intensity Difference')
legend_elements2 = [(Line2D([], [], linestyle='-', linewidth=1, color='#FF5380',
                            label='Cluster B Mean Subtracted from Cluster C Mean'))]
ax2.legend(handles=legend_elements2, loc='lower left', bbox_to_anchor=(0.0, 1.04), borderaxespad=0, ncol=2, fontsize=10)
fig2.show()

# overlaid mean spectral differences
fig3, ax3 = plt.subplots(1, 1)
fig3.subplots_adjust(left=0.15, right=0.95, top=0.8, bottom=0.1, wspace=0.2, hspace=0.2)
fig3.suptitle("Cluster Mean Subtractions", fontsize=14)
for i in range(len(cluster_a_subtraction)):
    ax3.plot(x_data[0, :], cluster_a_subtraction[i], color='#FF5380', linestyle='-', label='cluster_a_subtraction')
for i in range(len(cluster_b_subtraction)):
    ax3.plot(x_data[0, :], cluster_b_subtraction[i], color='#00FF00', linestyle='--', label='cluster_b_subtraction')
ax3.set(xlabel='Wavenumber (cm$^{-1}$)', ylabel='Intensity Difference')
legend_elements3 = [(Line2D([], [], linestyle='-', linewidth=1, color='#FF5380',
                            label='Cluster A Mean Subtracted from Cluster C Mean')),
                    (Line2D([], [], linestyle='--', linewidth=1, color='#00FF00',
                            label='Cluster B Mean Subtracted from Cluster C Mean'))]
ax3.legend(handles=legend_elements3, loc='lower left', bbox_to_anchor=(0.0, 1.04), borderaxespad=0, ncol=1, fontsize=10)
fig3.show()