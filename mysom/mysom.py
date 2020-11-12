"""

don't forget to look at time complexity - scalene, scalene, scalene, scaleeeeeeeene

"""

import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import kde
from sklearn import preprocessing
from minisom import MiniSom, _build_iteration_indexes, _wrap_index__in_verbose, fast_norm, asymptotic_decay


class MySom(MiniSom):
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5, random_seed=None):
        super().__init__(x, y, input_len, sigma=1.0, learning_rate=0.5, random_seed=None)
        # SOM parameters
        self.x = x
        self.y = y
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.decay_function = asymptotic_decay
        self.neighborhood_function = 'gaussian'
        self.topology = 'rectangular'
        self.activation_distance = 'euclidean'
        self.random_seed = int(random_seed)
        self.som = None

        # normalised data
        self.nydata = None

        # plot variables
        self.target = None
        self.t = None
        self.markers = []
        self.colours = []
        self.labels = []
        self.unlabelled_values = []


    def frobenius_norm(self, ydata):
        """normalise data by dividing each column by the Frobenius norm"""
        self.nydata = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, ydata)
        return self.nydata


    def scikit_norm(self, ydata):
        """normalise data using scikit learn preprocessing.scale function"""
        self.nydata = preprocessing.scale(ydata.T).T
        return self.nydata


    def make_som(self, train_iter):
        """implement MiniSom via mysom and allow manipulation of key parameters"""
        self.som = MiniSom(self.x, self.y, self.input_len, self.sigma, self.learning_rate, self.decay_function,
                      self.neighborhood_function, self.topology, self.activation_distance, self.random_seed)
        self.som.random_weights_init(self.nydata)
        print('Training SOM...')
        self.som.train_random(self.nydata, train_iter)
        print('Training complete!')
        return self.som


    def make_labels(self, y_path, label_list, marker_list, colour_list):
        """generate labels for SOM based on input data with subdivisions (x1, x2, x3 etc.)"""
        self.target = np.genfromtxt(y_path, delimiter=',', usecols=(0), dtype=str)
        self.t = []
        for v in self.target:  # iterate through values in self.target
            if any(x in v for x in label_list):  # any substring in label_list matches self.target string
                for i in label_list:  # for each matching item
                    if i in v:
                        self.t.append(label_list.index(i))  # append item index
                    elif i not in v:  # for each non-matching item
                        pass  # do nothing
            else:  # if no substring in label_list matches self.target string
                self.t.append('unlabelled')  # append unlabelled
        np.array(self.t)
        self.unlabelled_values = [i for i, v in enumerate(self.t) if v == 'unlabelled']  # returns list of indexes of values not in label_list
        if self.unlabelled_values:
            warnings.warn("Some values are unlabelled")
        self.markers = [marker for marker in marker_list]
        self.colours = [colour for colour in colour_list]
        self.labels = [label for label in label_list]
        return self.target, self.t, self.markers, self.colours, self.labels


    def plot_som_umatrix(self, figpath, datestr):
        """plot distance map (u-matrix) of SOM and overlay markers from mapped sample vectors"""
        # initialise figure canvas adn single axes
        figumatrix, ax1 = plt.subplots(1, 1)
        figumatrix.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)  # set whitespace around figure edges and space between subplots
        figumatrix.suptitle("Self Organising Map U-Matrix \n and Overlaid Input Data", fontsize=14)

        # fill in axes with SOM and overlaid input data
        ax1.pcolor(self.som.distance_map().T, cmap='Blues', alpha=1.0)  # plot transposed SOM distances in on matrix and set colormap
        for cnt, xx in enumerate(self.nydata):
            bmu = self.som.winner(xx)  # calculate BMU for sample
            ax1.plot(bmu[0] + 0.5, bmu[1] + 0.5, self.markers[self.t[cnt]],
                     markerfacecolor=self.colours[self.t[cnt]], markeredgecolor=self.colours[self.t[cnt]],
                     markersize=6, markeredgewidth=2)  # place marker on winning SOM node for sample xx
        ax1.axis([0, self.som._weights.shape[0], 0, self.som._weights.shape[1]])

        # add colorbar to figure
        divider1 = make_axes_locatable(ax1)
        ax1_cb = divider1.new_horizontal(size=0.3, pad=0.1)
        cb1 = colorbar.ColorbarBase(ax1_cb, cmap=cm.Blues, orientation='vertical', alpha=1.0)
        cb1.ax.get_yaxis().labelpad = 16
        cb1.ax.set_ylabel('Distance from Neurons in the Neighbourhood', rotation=270, fontsize=10.5)
        figumatrix.add_axes(ax1_cb)

        # add k legend elements using proxy artists where k is number of labels in label_list
        # generate legend elements with labels, markers, and colours defined in their script file lists
        legend_elements = []
        for i in range(len(self.labels)):
            legend_elements.append(Line2D([], [], marker=self.markers[i], color=self.colours[i], label=self.labels[i],
                            markerfacecolor=self.colours[i], markersize=8, markeredgewidth=2,
                            linestyle='None', linewidth=0))
        ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.0, 1.12),
                   borderaxespad=0, ncol=len(legend_elements), fontsize=10)

        figumatrix.show()
        figumatrix.savefig(figpath / f'{datestr}umatrix_sigma_{self.sigma}_learning-rate_'
                                     f'{self.learning_rate}_random-seed{self.random_seed}.eps', format='eps')


    def plot_som_scatter(self, figpath, datestr):
        """plot distance map (u-matrix) of SOM and scatter chart of markers representing co-ordinates of
        winning neurons across map with jitter to avoid overlap within cells"""
        # initialise figure canvas and single axes
        figscatter, ax1 = plt.subplots(1, 1)
        figscatter.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)  # set whitespace around figure edges and space between subplots
        figscatter.suptitle("Self Organising Map U-Matrix \n and Overlaid Scatterer Input Data", fontsize=14)

        # fill in axes with SOM and overlaid scatter data
        ax1.pcolor(self.som.distance_map().T, cmap='Blues', alpha=1.0)  # plot transposed SOM distances in one matrix and set colormap with reduced opacity
        ax1.grid()  # print grid in bold over background
        w_x, w_y = zip(*[self.som.winner(d) for d in self.nydata])  # get x an y variables
        w_x = np.array(w_x)  # convert x variables into np array
        w_y = np.array(w_y)  # convert y variables into np array
        for c in np.unique(self.t):  # plot scatter plot for sample
            idx_t = self.t == c
            ax1.scatter(w_x[idx_t] + .5 + (np.random.rand(np.sum(idx_t)) - .5) * .5,
                        w_y[idx_t] + .5 + (np.random.rand(np.sum(idx_t)) - .5) * .5, s=50, c=self.colours[c])

        # add colorbar to figure
        divider1 = make_axes_locatable(ax1)
        ax1_cb = divider1.new_horizontal(size=0.3, pad=0.1)
        cb1 = colorbar.ColorbarBase(ax1_cb, cmap=cm.Blues, orientation='vertical', alpha=1.0)
        cb1.ax.get_yaxis().labelpad = 16
        cb1.ax.set_ylabel('Distance from Neurons in the Neighbourhood', rotation=270, fontsize=10.5)
        figscatter.add_axes(ax1_cb)

        # add k legend elements using proxy artists where k is number of labels in label_list
        # generate legend elements with labels, markers, and colours defined in their script file lists
        legend_elements = []
        for i in range(len(self.labels)):
            legend_elements.append(Line2D([], [], marker=self.markers[i], color=self.colours[i], label=self.labels[i],
                                          markerfacecolor=self.colours[i], markersize=8, markeredgewidth=2,
                                          linestyle='None', linewidth=0))
        ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.0, 1.12),
                   borderaxespad=0, ncol=len(legend_elements), fontsize=10)

        figscatter.show()
        figscatter.savefig(figpath / f'{datestr}scatter_sigma_{self.sigma}_learning-rate_'
                                     f'{self.learning_rate}_random-seed{self.random_seed}.eps', format='eps')


    def plot_neuron_activation_frequency(self, figpath, datestr):
        """plot distance map (u-matrix) of SOM shaded to represnet frequency of neuron activation"""
        figneuract, ax1 = plt.subplots(1, 1)
        figneuract.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)  # set whitespace around figure edges and space between subplots
        figneuract.suptitle("Self Organising Map \n Neuron Activation Frequency", fontsize=14)

        # fill in axes with frequency of SOM neuron activation
        frequencies = self.som.activation_response(self.nydata)  # generate frequency of neuron activation
        ax1.pcolor(frequencies.T, cmap='Blues', alpha=1.0)  # plot tramsposed SOM frequencies in one matrix and set colourmap

        # add colorbar to figure
        divider1 = make_axes_locatable(ax1)
        ax1_cb = divider1.new_horizontal(size=0.3, pad=0.1)
        norm1 = mpl.colors.Normalize(vmin=np.min(frequencies), vmax=np.max(frequencies))  # define range for colorbar based on frequencies
        cb1 = colorbar.ColorbarBase(ax=ax1_cb, cmap=cm.Blues, norm=norm1, alpha=1.0, orientation='vertical')
        cb1.ax.get_yaxis().labelpad = 16
        cb1.ax.set_ylabel('Frequency of Neuron Activation', rotation=270, fontsize=10.5)
        figneuract.add_axes(ax1_cb)

        figneuract.show()
        figneuract.savefig(figpath / f'{datestr}neuron-activation_{self.sigma}_learning-rate_'
                                     f'{self.learning_rate}_random-seed{self.random_seed}.eps', format='eps')


    def plot_density_function(self, figpath, datestr):
        """g"""
        figdensity, ax1 = plt.subplots(1, 1)
        figdensity.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)  # set whitespace around figure edges and space between subplots
        figdensity.suptitle("Self Organising Map \n Density Plot", fontsize=14)

        # fill in axes with SOM and generate overlaid scatter data
        ax1.pcolor(self.som.distance_map().T, cmap='Blues', alpha=1.0)  # plot transposed SOM distances in one matrix and set colormap with reduced opacity
        ax1.grid()  # print grid in bold over background
        w_x, w_y = zip(*[self.som.winner(d) for d in self.nydata])  # get x an y variables
        w_x = np.array(w_x)  # convert x variables into np array
        w_y = np.array(w_y)  # convert y variables into np array

        # generate density plot data
        nbins=300  # number of bins
        k = kde.gaussian_kde([w_x, w_y])
        xi, yi = np.mgrid[w_x.min() : w_x.max() : nbins*1j, w_y.min() : w_y.max() : nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        # generate plot
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape))

        figdensity.show()
        figdensity.savefig(figpath / f'{datestr}density_sigma{self.sigma}_learning-rate_'
                                     f'{self.learning_rate}_random-seed{self.random_seed}.eps', format='eps')


    def plot_errors(self, max_iter, figpath, datestr):
        """plot quantisation and topographic error of SOM at each iteration step
        this analysis can help to understand training and to estimate optimum number of iterations"""
        # tell console training is in progress
        print('Calculating errors...')

        # Calculate errors for each iteration of SOM
        q_error = []
        t_error = []
        for i in range(max_iter):
            rand_i = np.random.randint(len(self.nydata))
            self.som.update(self.nydata[rand_i], self.som.winner(self.nydata[rand_i]), i, max_iter)
            q_error.append(self.som.quantization_error(self.nydata))
            t_error.append(self.som.topographic_error(self.nydata))

        # tell console error calculation is complete
        print('Error calculation complete')

        # initialise figure canvas and two axes
        figerrors, (ax1, ax2) = plt.subplots(2, 1)
        figerrors.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)  # set whitespace around figure edges and space between subplots
        figerrors.suptitle("Quantisation and Topographic Error of \n Self Organising Map Method", fontsize=14)

        # fill in axes 1 with quantisation error and axes 2 with topographic error
        ax1.plot(np.arange(max_iter), q_error, color='#00BFFF', label='Quantisation Error')
        ax1.set(xlabel='Iteration', ylabel='Error')
        ax2.plot(np.arange(max_iter), t_error, color='#FFA500', label='Topographic Error')
        ax2.set(xlabel='Iteration', ylabel='Error')

        # add legend using proxy artists
        legend_elements_1 = [Line2D([], [], linestyle='-', linewidth=1, color='#00BFFF', label='Quantisation Error')]
        legend_elements_2 = [Line2D([], [], linestyle='-', linewidth=1, color='#FFA500', label='Topographic Error')]
        ax1.legend(handles=legend_elements_1, loc='upper right', bbox_to_anchor=(0.99, 0.97),
                   borderaxespad=0, ncol=len(legend_elements_1), fontsize=10)
        ax2.legend(handles=legend_elements_2, loc='upper right', bbox_to_anchor=(0.99, 0.97),
                   borderaxespad=0, ncol=len(legend_elements_2), fontsize=10)

        figerrors.show()
        figerrors.savefig(figpath / f'{datestr}errors_{self.sigma}_learning-rate_'
                                     f'{self.learning_rate}_random-seed{self.random_seed}.eps', format='eps')
