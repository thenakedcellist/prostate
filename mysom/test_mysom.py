import pytest
import numpy as np
from pathlib import Path

from mysom.mysom import MySom


@pytest.fixture
def fix_matrix():
    matrix = np.array([[1, 2, 3, 4],  # row total 30
                       [5, 6, 7, 8],  # row total 174
                       [9, 10, 11, 12]])  # row total 446
    return matrix

@pytest.fixture
def fix_som(fix_matrix):
    som = MySom(x=6, y=5, input_len=4, sigma=1, learning_rate=1, random_seed=1)
    som.som_setup((1, 2, 3, 4), fix_matrix, 'test_source.txt', ('Huey', 'Dewey', 'Louie'), ('o', '+', 'd'), ('r', 'g', 'b'))
    som.nydata = fix_matrix
    som.train_som(10)
    som._weights = np.zeros((5, 3, 4))  # all zero weights
    som._weights[1, 1] = 2.0
    som._weights[4, 2] = 5.0
    return som

class TestNormalisation:

    def test_frobenius_norm_normalisation(self, fix_som):
        """normalise by dividing values by frobenius mean across columns"""
        d = MySom.frobenius_norm_normalisation(fix_som)
        np.testing.assert_array_almost_equal(d, [[0.182574, 0.365148, 0.547723, 0.730297],
                                                 [0.379049, 0.454859, 0.530669, 0.606478],
                                                 [0.426162, 0.473514, 0.520865, 0.568216]])

    def test_scikit_normalisation(self, fix_som):
        """transform data to have zero mean and unit variance across columns"""
        d = MySom.scikit_normalisation(fix_som)
        np.testing.assert_array_almost_equal(d.mean(axis=1), [0.0, 0.0, 0.0])  # each row has zero mean
        np.testing.assert_array_almost_equal(d.std(axis=1), [1.0, 1.0, 1.0])  # each row has unit variance


class TestSomBuilding:

    def test_som_weights(self, fix_som):
        print(fix_som._weights)

    def test_correct_labels(self, fix_som):
        labels = ['Huey', 'Dewey', 'Louie']
        markers = ['o', '+', 'd']
        colours = ['r', 'g', 'b']
        fix_som.som_setup((1, 2, 3, 4), fix_matrix, 'test_source.txt', labels, markers, colours)
        print(fix_som.t)

    def test_some_incorrect_labels(self, fix_som):
        labels = ['Hewey', 'Dewey', 'Lewey']
        markers = ['o', '+', 'd']
        colours = ['r', 'g', 'b']
        fix_som.som_setup((1, 2, 3, 4), fix_matrix, 'test_source.txt', labels, markers, colours)
        print(fix_som.t)

    def test_all_incorrect_labels(self, fix_som):
        labels = ['Hewey', 'Dewie', 'Lewey']
        markers = ['o', '+', 'd']
        colours = ['r', 'g', 'b']
        fix_som.som_setup((1, 2, 3, 4), fix_matrix, 'test_source.txt', labels, markers, colours)
        print(fix_som.t)

    def test_no_labels(self, fix_som):
        labels = []
        markers = ['o', '+', 'd']
        colours = ['r', 'g', 'b']
        fix_som.som_setup((1, 2, 3, 4), fix_matrix, 'test_source.txt', labels, markers, colours)
        print(fix_som.t)

    def test_blinded_data(self, fix_som):
        labels = ['Blinded Data']
        markers = ['o', '+', 'd']
        colours = ['r', 'g', 'b']
        fix_som.som_setup((1, 2, 3, 4), fix_matrix, 'test_source.txt', labels, markers, colours)
        print(fix_som.t)

    def test_som_weights(self, fix_som):
        print(fix_som._weights)
        #assert fix_som._weights[1, 1] == 2.0
        #assert fix_som._weights[2, 3] == 5.0
        #assert fix_som._weights[4, 4] == 0.0


class TestOutputGraphs:

    def test_som_umatrix_show_input(self, fix_som, tmp_path):
        fix_som.target = ['Huey', 'Dewey', 'Louie']
        fix_som.t = [0, 1, 2]
        fix_som.labels = ['Huey', 'Dewey', 'Louie']
        fix_som.markers = ['o', '+', 'd']
        fix_som.colours = ['r', 'g', 'b']
        fix_som.plot_som_umatrix(tmp_path, "ducks", showinput=True, showinactivenodes=False, onlyshow=True)

    def test_som_umatrix_hide_input(self, fix_som, tmp_path):
        fix_som.target = ['Huey', 'Dewey', 'Louie']
        fix_som.t = [0, 1, 2]
        fix_som.labels = ['Huey', 'Dewey', 'Louie']
        fix_som.markers = ['o', '+', 'd']
        fix_som.colours = ['r', 'g', 'b']
        fix_som.plot_som_umatrix(tmp_path, "ducks", showinput=False, showinactivenodes=False, onlyshow=True)

    def test_som_umatrix_show_inactive_nodes(self, fix_som, tmp_path):
        fix_som.target = ['Huey', 'Dewey', 'Louie']
        fix_som.t = [0, 1, 2]
        fix_som.labels = ['Huey', 'Dewey', 'Louie']
        fix_som.markers = ['o', '+', 'd']
        fix_som.colours = ['r', 'g', 'b']
        fix_som.plot_som_umatrix(tmp_path, "ducks", showinput=False, showinactivenodes=True, onlyshow=True)

    def test_som_scatter(self, fix_som, tmp_path):
        fix_som.target = ['Huey', 'Dewey', 'Louie']
        fix_som.t = [0, 1, 2]
        fix_som.labels = ['Huey', 'Dewey', 'Louie']
        fix_som.markers = ['o', '+', 'd']
        fix_som.colours = ['r', 'g', 'b']
        fix_som.plot_som_scatter(tmp_path, "ducks", onlyshow=True)

    def test_neuron_freq(self, fix_som, tmp_path):
        fix_som.plot_neuron_activation_frequency(tmp_path, "ducks", onlyshow=True)

    def test_density_func(self, fix_som, tmp_path):
        fix_som.plot_density_function(tmp_path, "ducks", onlyshow=True)

    def test_plot_errors(self, fix_som, tmp_path):
        fix_som.plot_errors(10, tmp_path, "ducks", onlyshow=True)
