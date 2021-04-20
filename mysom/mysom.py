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
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=None):
        super().__init__(x, y, input_len, sigma=1.0, learning_rate=0.5, topology='rectangular', random_seed=None)
        # SOM parameters
        self.x = x
        self.y = y
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.decay_function = asymptotic_decay
        self.neighborhood_function = 'gaussian'
        self.topology = topology
        self.activation_distance = 'euclidean'
        self.random_seed = int(random_seed)
        self.som = None

        # input data
        self.xdata = None
        self.ydata = None
        self.nydata = None

        # plot variables
        self.target = None
        self.t = None
        self.markers = []
        self.colours = []
        self.labels = []
        self.unlabelled_values = []

        # error calculation
        self.err_som = None
        self.quantisation_err = None
        self.topographic_err = None

    def som_setup(self, xdata, ydata, y_path, label_list, marker_list, colour_list):
        """setup input data and parameters for SOM, and generate labels, colours, and markers"""
        self.som = MiniSom(self.x, self.y, self.input_len, self.sigma, self.learning_rate, self.decay_function,
                           self.neighborhood_function, self.topology, self.activation_distance, self.random_seed)
        self.xdata = xdata
        self.ydata = ydata
        self.target = np.genfromtxt(y_path, delimiter=',', usecols=(0), dtype=str)
        self.t = []

        # creates ndarray t of zeros for blinded data
        if label_list == ['Blinded Data']:
            self.t = np.zeros(self.ydata.shape[0], dtype=int)  # all values set to 0
            # print warning message to indicate data are blinded
            warnings.warn("Data are blinded")

        # creates list t of index values of labels in label_list corresponding to observations
        else:
            for v in self.target:  # iterate through values in self.target
                if any(x in v for x in label_list):  # any substring in label_list matches self.target string
                    for i in label_list:  # for each matching item
                        if i in v:
                            self.t.append(label_list.index(i))
                            break  # append item index
        # appends [unlabelled] to list t in index of observation which does not match any label in label_list
                else:  # if no substring in label_list matches self.target string
                    self.t.append('unlabelled')  # append unlabelled
        # generate list of indices of observations which are unlabelled
        self.unlabelled_values = [i for i, v in enumerate(self.t) if v == 'unlabelled']

        # print warning message indicating how many values are labelled
        if not self.unlabelled_values and label_list != ['Blinded Data']:
            warnings.warn("All values are labelled")
        if len(self.unlabelled_values) == len(self.t):
            warnings.warn("All values are unlabeled")
        elif self.unlabelled_values:
            warnings.warn("Some values are unlabelled")

        # convert t into ndarray
        self.t = np.array(self.t)

        # add attributes to mysom instance
        self.markers = [marker for marker in marker_list]
        self.colours = [colour for colour in colour_list]
        self.labels = [label for label in label_list]
        return self.som, self.xdata, self.ydata, self.target, self.t, self.markers, self.colours, self.labels

    def remove_observations_from_input_data(self, removal_list):
        """parse list of indices and remove them from som y data"""
        for i in removal_list:
            self.ydata = np.delete(self.ydata, i, 0)
            self.target = np.delete(self.target, i, 0)
            self.t = np.delete(self.t, i, 0)
        return self.ydata, self.target, self.t

    def frobenius_norm_normalisation(self):
        """normalise data by dividing each column by the Frobenius norm"""
        self.nydata = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, self.ydata)
        return self.nydata

    def scikit_normalisation(self):
        """normalise data using scikit learn preprocessing.scale function"""
        self.nydata = preprocessing.scale(self.ydata.T).T
        return self.nydata

    def train_som(self, train_iter):
        """use MiniSom methods via mysom to train SOM"""
        self.som.random_weights_init(self.nydata)
        print('Training SOM...')
        self.som.train_random(self.nydata, train_iter)
        print('Training complete!')
        self.quantisation_err = self.som.quantization_error(self.nydata)
        self.topographic_err = self.som.topographic_error(self.nydata)
        return self.som, self.quantisation_err, self.topographic_err

    def plot_som_umatrix(self, figpath, datestr, showinput=False, showinactivenodes=False, onlyshow=False, eps=False):
        """plot distance map (u-matrix) of SOM and overlay markers from mapped sample vectors"""
        # initialise figure canvas and single axes
        figumatrix, ax1 = plt.subplots(1, 1)
        # set whitespace around figure edges and space between subplots
        figumatrix.subplots_adjust(left=0.125, right=0.9, top=0.8, bottom=0.1, wspace=0.2, hspace=0.2)
        figumatrix.suptitle("Self Organising Map U-Matrix", fontsize=14)
        ax1.locator_params(axis='both', integer=True)
        ax1.set_aspect('equal')

        # fill in axes with SOM and overlaid input data
        # plot transposed SOM distances in on matrix and set colormap
        ax1.pcolor(self.som.distance_map().T, cmap='Blues_r', alpha=1.0)

        if showinput:
            # overlay input data
            for cnt, xx in enumerate(self.nydata):
                bmu = self.som.winner(xx)  # calculate BMU for sample
                ax1.plot(bmu[0] + 0.5, bmu[1] + 0.5, self.markers[self.t[cnt]],
                         markerfacecolor=self.colours[self.t[cnt]], markeredgecolor=self.colours[self.t[cnt]],
                         markersize=(30/self.x), markeredgewidth=2)  # place marker on winning SOM node for sample xx
            ax1.axis([0, self.som._weights.shape[0], 0, self.som._weights.shape[1]])
        else:
            pass

        if showinactivenodes:
            pass
        else:
            pass
        # TODO cross to show which neurons not activated (set of all points, which not in som.winner(), plot these as above

        # add colorbar to figure
        divider1 = make_axes_locatable(ax1)
        ax1_cb = divider1.new_horizontal(size=0.3, pad=0.1)
        cb1 = colorbar.ColorbarBase(ax1_cb, cmap=cm.Blues_r, orientation='vertical', alpha=1.0)
        cb1.ax.get_yaxis().labelpad = 16
        cb1.ax.set_ylabel('Distance from Nodes in the Neighbourhood', rotation=270, fontsize=10.5)
        figumatrix.add_axes(ax1_cb)

        # add k legend elements using proxy artists where k is number of labels in label_list
        # generate legend elements with labels, markers, and colours defined in their script file lists
        legend_elements = []
        for i in range(len(self.labels)):
            legend_elements.append(Line2D([], [], marker=None, color=None, label=None,
                                          markerfacecolor=None, markersize=None, markeredgewidth=None,
                                          linestyle='None', linewidth=None))
        ax1.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.0, 1.04),
                   borderaxespad=0, ncol=3, fontsize=None, frameon=False)

        figumatrix.show()
        if onlyshow:
            pass
        elif eps:
            figumatrix.savefig(figpath / 'eps' / f'{datestr}_som-umatrix_x_{self.x}_y_{self.y}_sigma_{self.sigma}'
                                                 f'_learning-rate_{self.learning_rate}'
                                                 f'_random-seed_{self.random_seed}.eps', format='eps')
        else:
            figumatrix.savefig(figpath / 'png' / f'{datestr}_som-umatrix_x_{self.x}_y_{self.y}_sigma_{self.sigma}'
                                                 f'_learning-rate_{self.learning_rate}'
                                                 f'_random-seed_{self.random_seed}.png', format='png')

    def plot_som_scatter(self, figpath, datestr, onlyshow=False, eps=False):
        """plot distance map (u-matrix) of SOM and scatter chart of markers representing co-ordinates of
        winning neurons across map with jitter to avoid overlap within cells"""
        # initialise figure canvas and single axes
        figscatter, ax1 = plt.subplots(1, 1)
        # set whitespace around figure edges and space between subplots
        figscatter.subplots_adjust(left=0.125, right=0.9, top=0.8, bottom=0.1, wspace=0.2, hspace=0.2)
        figscatter.suptitle("Self Organising Map U-Matrix with Overlaid Scatter Input Data", fontsize=14)
        ax1.locator_params(axis='both', integer=True)
        ax1.set_aspect('equal')

        # fill in axes with SOM and overlaid scatter data
        ax1.pcolor(self.som.distance_map().T, cmap='Blues_r',
                   alpha=1.0)  # plot transposed SOM distances in one matrix and set colormap with reduced opacity
        w_x, w_y = zip(*[self.som.winner(d) for d in self.nydata])  # get x an y variables
        w_x = np.array(w_x)  # convert x variables into np array
        w_y = np.array(w_y)  # convert y variables into np array
        for c in np.unique(self.t):  # plot scatter plot for sample
            idx_t = self.t == c
            ax1.scatter(w_x[idx_t] + .5 + (np.random.rand(np.sum(idx_t)) - .5) * .5,
                        w_y[idx_t] + .5 + (np.random.rand(np.sum(idx_t)) - .5) * .5,
                        s=(30/self.x)**2, c=self.colours[c], marker=self.markers[c])

        # add colorbar to figure
        divider1 = make_axes_locatable(ax1)
        ax1_cb = divider1.new_horizontal(size=0.3, pad=0.1)
        cb1 = colorbar.ColorbarBase(ax1_cb, cmap=cm.Blues_r, orientation='vertical', alpha=1.0)
        cb1.ax.get_yaxis().labelpad = 16
        cb1.ax.set_ylabel('Distance from Nodes in the Neighbourhood', rotation=270, fontsize=10.5)
        figscatter.add_axes(ax1_cb)

        # add k legend elements using proxy artists where k is number of labels in label_list
        # generate legend elements with labels, markers, and colours defined in their script file lists
        legend_elements = []
        for i in range(len(self.labels)):
            legend_elements.append(Line2D([], [], marker=self.markers[i], color=self.colours[i], label=self.labels[i],
                                          markerfacecolor=self.colours[i], markersize=8, markeredgewidth=2,
                                          linestyle='None', linewidth=0))
        ax1.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.0, 1.04),
                   borderaxespad=0, ncol=3, fontsize=10)

        figscatter.show()
        if onlyshow:
            pass
        elif eps:
            figscatter.savefig(figpath / 'eps' / f'{datestr}_scatter-plot_x_{self.x}_y_{self.y}_sigma_{self.sigma}'
                                                 f'_learning-rate_{self.learning_rate}'
                                                 f'_random-seed_{self.random_seed}.eps', format='eps')
        else:
            figscatter.savefig(figpath / 'png' / f'{datestr}_scatter-plot_x_{self.x}_y_{self.y}_sigma_{self.sigma}'
                                                 f'_learning-rate_{self.learning_rate}'
                                                 f'_random-seed_{self.random_seed}.png', format='png')

    def plot_node_activation_frequency(self, figpath, datestr, onlyshow=False, eps=False):
        """plot distance map (u-matrix) of SOM shaded to represnet frequency of neuron activation"""
        fignodeact, ax1 = plt.subplots(1, 1)
        # set whitespace around figure edges and space between subplots
        fignodeact.subplots_adjust(left=0.125, right=0.9, top=0.8, bottom=0.1, wspace=0.2, hspace=0.2)
        fignodeact.suptitle("Self Organising Map Node Activation Frequency", fontsize=14)
        ax1.locator_params(axis='both', integer=True)
        ax1.set_aspect('equal')

        # fill in axes with frequency of SOM neuron activation
        frequencies = self.som.activation_response(self.nydata)  # generate frequency of neuron activation
        ax1.pcolor(frequencies.T, cmap='Blues',
                   alpha=1.0)  # plot tramsposed SOM frequencies in one matrix and set colourmap

        # add colorbar to figure
        divider1 = make_axes_locatable(ax1)
        ax1_cb = divider1.new_horizontal(size=0.3, pad=0.1)
        norm1 = mpl.colors.Normalize(vmin=np.min(frequencies),
                                     vmax=np.max(frequencies))  # define range for colorbar based on frequencies
        cb1 = colorbar.ColorbarBase(ax=ax1_cb, cmap=cm.Blues, norm=norm1, alpha=1.0, orientation='vertical')
        cb1.ax.get_yaxis().labelpad = 16
        cb1.ax.set_ylabel('Frequency of Node Activation', rotation=270, fontsize=10.5)
        fignodeact.add_axes(ax1_cb)

        # empty legend box to keep constraints uniform across plots
        legend_elements = []
        for i in range(len(self.labels)):
            legend_elements.append(Line2D([], [], marker=None, color=None, label=None,
                                          markerfacecolor=None, markersize=None, markeredgewidth=None,
                                          linestyle='None', linewidth=None))
        ax1.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.0, 1.04),
                   borderaxespad=0, ncol=3, fontsize=None, frameon=False)

        fignodeact.show()
        if onlyshow:
            pass
        elif eps:
            fignodeact.savefig(figpath / 'eps' / f'{datestr}_node-freq_{self.x}_y_{self.y}_sigma_{self.sigma}'
                                                 f'_learning-rate_{self.learning_rate}'
                                                 f'_random-seed_{self.random_seed}.eps', format='eps')
        else:
            fignodeact.savefig(figpath / 'png' / f'{datestr}_node-freq_{self.x}_y_{self.y}_sigma_{self.sigma}'
                                                 f'_learning-rate_{self.learning_rate}'
                                                 f'_random-seed_{self.random_seed}.png', format='png')

    def plot_density_function(self, figpath, datestr, onlyshow=False, eps=False):
        """plot density of neuron activation across SOM"""
        figdensity, ax1 = plt.subplots(1, 1)
        # set whitespace around figure edges and space between subplots
        figdensity.subplots_adjust(left=0.125, right=0.9, top=0.8, bottom=0.1, wspace=0.2, hspace=0.2)
        figdensity.suptitle("Self Organising Map Density Plot", fontsize=14)
        ax1.locator_params(axis='both', integer=True)
        ax1.set_aspect('equal')

        # fill in axes with SOM and generate overlaid scatter data
        w_x, w_y = zip(*[self.som.winner(d) for d in self.nydata])  # get x an y variables
        w_x = (np.array(w_x) + 0.5)  # convert x variables into np array with 0.5 added to each value
        w_y = (np.array(w_y) + 0.5)  # convert y variables into np array with 0.5 added to each value

        # generate density plot data
        nbins = 200  # number of bins
        k = kde.gaussian_kde([w_x, w_y])  # initiate gaussian kernel density estimate
        # generate regular grid of nbins x nbins over data, maximum value is inclusive
        xi, yi = np.mgrid[0: self.x: nbins * 1j, 0:  self.y: nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='Oranges')  # generate kde plot
        ax1.contour(xi, yi, zi.reshape(xi.shape), cmap='Blues')  # add contour lines

        # add colorbar to figure
        frequencies = self.som.activation_response(self.nydata)  # generate frequency of neuron activation
        divider1 = make_axes_locatable(ax1)
        ax1_cb1 = divider1.new_horizontal(size=0.3, pad=0.1)
        norm1 = mpl.colors.Normalize(vmin=np.min(frequencies),
                                     vmax=np.max(frequencies))  # define range for colorbar based on frequencies
        cb1 = colorbar.ColorbarBase(ax=ax1_cb1, cmap=cm.Oranges, norm=norm1, alpha=1.0, orientation='vertical')
        cb1.ax.get_yaxis().labelpad = 16
        cb1.ax.set_ylabel('Density of Node Activation', rotation=270, fontsize=10.5)
        figdensity.add_axes(ax1_cb1)

        # empty legend box to keep constraints uniform across plots
        legend_elements = []
        for i in range(len(self.labels)):
            legend_elements.append(Line2D([], [], marker=None, color=None, label=None,
                                          markerfacecolor=None, markersize=None, markeredgewidth=None,
                                          linestyle='None', linewidth=None))
        ax1.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.0, 1.04),
                   borderaxespad=0, ncol=3, fontsize=None, frameon=False)

        figdensity.show()
        if onlyshow:
            pass
        elif eps:
            figdensity.savefig(figpath / 'eps' / f'{datestr}_density-est_x_{self.x}_y_{self.y}_sigma_{self.sigma}'
                                                 f'_learning-rate_{self.learning_rate}'
                                                 f'_random-seed_{self.random_seed}.eps', format='eps')
        else:
            figdensity.savefig(figpath / 'png' / f'{datestr}_density-est_x_{self.x}_y_{self.y}_sigma_{self.sigma}'
                                                 f'_learning-rate_{self.learning_rate}'
                                                 f'_random-seed_{self.random_seed}.png', format='png')

    def plot_errors(self, max_iter, figpath, datestr, onlyshow=False, eps=False):
        """plot quantisation and topographic error of SOM at each iteration step
        this analysis can help to understand training and to estimate optimum number of iterations
        only available for rectangular topology"""

        # copy som to err-som without altering som
        self.err_som = self.som

        # tell console training is in progress
        print('Calculating errors...')

        # Calculate errors for each iteration of SOM
        q_error = []
        t_error = []
        for i in range(max_iter):
            rand_i = np.random.randint(len(self.nydata))
            self.err_som.update(self.nydata[rand_i], self.err_som.winner(self.nydata[rand_i]), i, max_iter)
            q_error.append(self.err_som.quantization_error(self.nydata))
            t_error.append(self.err_som.topographic_error(self.nydata))

        # tell console error calculation is complete
        print('Error calculation complete')

        # initialise figure canvas and two axes
        figerrors, (ax1, ax2) = plt.subplots(2, 1)
        # set whitespace around figure edges and space between subplots
        figerrors.subplots_adjust(left=0.125, right=0.9, top=0.8, bottom=0.1, wspace=0.2, hspace=0.3)
        figerrors.suptitle("Quantisation and Topographic Error of Self Organising Map", fontsize=14)

        # fill in axes 1 with quantisation error and axes 2 with topographic error
        ax1.plot(np.arange(max_iter), q_error, color='#00BFFF', label='Quantisation Error')
        ax1.set(xlabel='Iteration', ylabel='Error')
        ax2.plot(np.arange(max_iter), t_error, color='#FFA500', label='Topographic Error')
        ax2.set(xlabel='Iteration', ylabel='Error')

        # add legend using proxy artists
        legend_elements = [Line2D([], [], linestyle='-', linewidth=1, color='#00BFFF', label='Quantisation Error'),
                          Line2D([], [], linestyle='-', linewidth=1, color='#FFA500', label='Topographic Error')]
        ax1.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.0, 1.04),
                         borderaxespad=0, ncol=2, fontsize=10)

        figerrors.show()
        if onlyshow:
            pass
        elif eps:
            figerrors.savefig(figpath / 'eps' / f'{datestr}_q-t-errors_x_{self.x}_y_{self.y}_sigma_{self.sigma}'
                                                f'_learning-rate_{self.learning_rate}'
                                                f'_random-seed_{self.random_seed}.eps', format='eps')
        else:
            figerrors.savefig(figpath / 'png' / f'{datestr}_q-t-errors_x_{self.x}_y_{self.y}_sigma_{self.sigma}'
                                                f'_learning-rate_{self.learning_rate}'
                                                f'_random-seed_{self.random_seed}.png', format='png')

    def plot_som_umatrix_hex(self, figpath, datestr):
        """plot SOM u-matrix for hexagonal topology SOM"""
        from matplotlib.patches import RegularPolygon
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib import cm, colorbar
        from matplotlib.lines import Line2D

        # initialise figure canvas and single axes
        figumatrixhex, ax1 = plt.subplots(1, 1)
        # set whitespace around figure edges and space between subplots
        figumatrixhex.subplots_adjust(left=0.125, right=0.9, top=0.8, bottom=0.1, wspace=0.2, hspace=0.2)
        figumatrixhex.suptitle("Self Organising Map U-Matrix", fontsize=14)
        ax1.set_aspect('equal')

        # get size, values, and weights from SOM
        xx, yy = self.som.get_euclidean_coordinates()
        umatrix = self.som.distance_map()
        weights = self.som.get_weights()

        # form hex grid
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                wy = yy[(i, j)] * 2 / np.sqrt(3) * 3 / 4
                hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95 / np.sqrt(3),
                                     facecolor=cm.Blues_r(umatrix[i, j]), alpha=1.0, edgecolor='gray')
                ax1.add_patch(hex)

        # calculate and plot BMU for sample
        for cnt, xx in enumerate(self.nydata):
            bmu = self.som.winner(xx)  # calculate BMU
            wx, wy = self.som.convert_map_to_euclidean(bmu)
            wy = wy * 2 / np.sqrt(3) * 3 / 4
            ax1.plot(wx, wy, self.markers[self.t[cnt]], markerfacecolor=self.colours[self.t[cnt]],
                     markeredgecolor=self.colours[self.t[cnt]], markersize=6,
                     markeredgewidth=2)  # place marker on winning position for sample xx

        # set x and y range of plot
        xrange = np.arange(weights.shape[0])
        yrange = np.arange(weights.shape[1])
        plt.xticks(xrange - .5, xrange)
        plt.yticks(yrange * 2 / np.sqrt(3) * 3 / 4, yrange)

        # add colorbar to figure
        divider1 = make_axes_locatable(ax1)
        ax1_cb = divider1.new_horizontal(size=0.3, pad=0.1)
        cb1 = colorbar.ColorbarBase(ax1_cb, cmap=cm.Blues_r, orientation='vertical', alpha=1.0)
        cb1.ax.get_yaxis().labelpad = 16
        cb1.ax.set_ylabel('Distance from Nodes in the Neighbourhood', rotation=270, fontsize=10.5)
        figumatrixhex.add_axes(ax1_cb)

        # add k legend elements using proxy artists where k is number of labels in label_list
        # generate legend elements with labels, markers, and colours defined in their script file lists
        legend_elements = []
        for i in range(len(self.labels)):
            legend_elements.append(Line2D([], [], marker=self.markers[i], color=self.colours[i], label=self.labels[i],
                                          markerfacecolor=self.colours[i], markersize=8, markeredgewidth=2,
                                          linestyle='None', linewidth=0))
        ax1.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.0, 1.04),
                   borderaxespad=0, ncol=3, fontsize=10)

        plt.show()

'''
------------------------------------------------------------------------------------------------------------------------
rewrite these so that they can be used in conjuction with winmap/labmap and spectrum plotter
test suite is paramount
likely as second module - spectral plotter


    def unpack_all_data_points_by_group(self):
        """generate lists of all data indices by group"""
        PNT2_all = []
        LNCaP_all = []
        for i, v in enumerate(self.t):
            if v == 0:
                PNT2_all.append(i)
            elif v == 1:
                LNCaP_all.append(i)
        return PNT2_all, LNCaP_all

    def unpack_points_of_interest_by_group(self, poi):
        """generate lists of data indices of points of interest by group"""
        # flatten list of points of interest
        flat_poi = [item for sublist in poi for item in sublist]
        # generate list of corresponding label values for points of interest
        labs_poi = [self.t[i] for i in flat_poi]
        # generate lists of point of interest indices by group
        PNT2_poi = []
        LNCaP_poi = []
        for i, v in enumerate(labs_poi):
            if v == 0:
                PNT2_poi.append(flat_poi[i])
            elif v == 1:
                LNCaP_poi.append(flat_poi[i])
        return PNT2_poi, LNCaP_poi

    def remove_poi_from_list(self, list_a, list_b):
        """takes two lists A and B, and removes the values of list B from list A"""
        concat_list = [v for v in list_a if v not in list_b]
        return concat_list

    def make_poi_array(self, input_list):
        """generate array of input data with indices from input list"""
        output_arr = self.ydata[np.ix_(input_list)]
        return output_arr

    def make_array_column_mean(self, input_arr):
        """calculate column mean of input array"""
        output_arr = input_arr.mean(axis=0)[None, :]
        return output_arr

    def plot_spectra_from_poi(self, cluster_PNT2_input_arr, cluster_LNCaP_input_arr, figpath, title_string, onlyshow=False):
        """plot average spectra for cluster by group"""
        fig, ax = plt.subplots(1, 1)
        # set whitespace around figure edges and space between subplots
        fig.subplots_adjust(left=0.125, right=0.9, top=0.8, bottom=0.1, wspace=0.2, hspace=0.2)
        fig.suptitle(title_string, fontsize=14)
        for i in range(len(cluster_PNT2_input_arr)):
            ax.plot(self.xdata[0, :], cluster_PNT2_input_arr[i], color='#FFA500', label='PNT2')
        for i in range(len(cluster_LNCaP_input_arr)):
            ax.plot(self.xdata[0, :], cluster_LNCaP_input_arr[i], color='g', label='LNCaP')
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
'''
