import pytest
import banbury_data_extractor as by
from banbury_data_extractor import ByTissueByAnimal
from pathlib import Path
import numpy as np

@pytest.fixture
def lens2():
    """returns ByTissueByAnimal instance of lens 2"""
    return ByTissueByAnimal("lens", 2)


@pytest.fixture
def lens3():
    """returns ByTissueByAnimal instance of lens 3"""
    return ByTissueByAnimal("lens", 3)


@pytest.fixture
def vitreous2():
    """returns ByTissueByAnimal instance of vitreous 2"""
    return ByTissueByAnimal("vitreous", 2)


def test_generate_x_data():
    """check that global x data are generated as a 1 x n 2d array"""
    # check that x labels are 1 x 1 2d array containing [wavenumber]
    assert by.wave_lab.shape[0] == 1  # check for exactly one row of data
    assert by.wave_lab.shape[1] == 1  # check for exactly one column of data
    assert by.wave_lab[0, 0] == "wavenumber"

    # check that x data are 1 x n 2d array which is not empty
    assert by.wavenumbers.shape[0] == 1  # check for exactly one row of data
    assert by.wavenumbers.shape[1] > by.wavenumbers.shape[0]  # check for more columns than rows
    assert by.wavenumbers.size != 0


def test_save_x_data():
    """check that x data csv files are saved to _data directory and three subdirectories"""
    by.save_x_data()
    assert Path('../../data/banbury/_data/by_tissue_by_animal/_wavenumbers.csv').exists() == True
    assert Path('../../data/banbury/_data/all_tissues_by_animal/_wavenumbers.csv').exists() == True
    assert Path('../../data/banbury/_data/by_tissue_all_animals/_wavenumbers.csv').exists() == True
    assert Path('../../data/banbury/_data/_wavenumbers.csv').exists() == True


def test_file_search(lens2):
    """check that banbury_data_extractor directory and tissue subdirectory for input tissue exist
    and that a non-empty dictionary is created"""
    lens2.file_search()
    assert lens2.file_dir.exists() == True
    assert len(lens2.file_dict) != 0


def test_generate_y_labels(lens2):
    """check y labels are generated as a 1 x n 2d array of file dictionary keys"""
    lens2.file_search()
    lens2.generate_y_labels()
    assert lens2.y_labels.shape[1] == 1  # check for exactly one column of data
    assert lens2.y_labels.shape[0] == len(lens2.file_dict)  # check for as many rows of data as entries in file_dict
    assert lens2.y_labels[0, 0] == "animal_2_lens_0"


def test_extract_y_data(lens2):
    """check x data are extracted from first file in directory, and that an array with data in rows is created"""
    lens2.file_search()
    lens2.extract_y_data()
    assert list(lens2.file_dict.values())[0] == '../../data/banbury_data_extractor/lens/eye_2_lens_fp_sub_0__X_7116.25__Y_-3658.15__Time_0__Zdata_4719.13__Zactual_4718.68__Zdifference_-0.445551__LTSignalUsed_3.txt'
    assert lens2.y_data.shape[0] == len(lens2.file_dict)  # check for as many rows as in file dictionary so none have been skipped
    assert lens2.y_data.shape[1] > lens2.y_data.shape[0]  # check for more columns than rows


def test_extract_y_data(lens2):
    """check y data are extracted from all files in directory, and that array with as many rows as dictionary elements
    and as many columns as x data is created"""
    lens2.file_search()
    lens2.extract_y_data()
    assert lens2.y_data.shape[0] == len(lens2.file_dict)
    assert lens2.y_data.shape[1] == lens2.x_data.shape[1]
    assert lens2.y_data[0,0] == 0.668831


def test_save_y_data(lens2):
    """check data are saved as csv within by_tissue_by_animal subdirectory"""
    lens2.do_all_the_things()
    lens2.save_y_data()
    assert Path('../../data/banbury/_data/by_tissue_by_animal/animal_2_lens.csv').exists() == True

def test_all_tissues_by_animal(lens2, vitreous2):
    """check animal number csv file produced
    in all_tissues_by_animal subdirectory
    containing data from all included tissue types """
    lens2.do_all_the_things()
    vitreous2.do_all_the_things()
    by.all_tissues_by_animal(2)
    path = Path('../../data/banbury/_data/all_tissues_by_animal/animal_2.csv')
    assert Path(path).exists() == True
    query = np.genfromtxt(path, dtype='U', delimiter=',', usecols=0)
    assert query[0] == "animal_2_lens_0"
    assert query[91] == "animal_2_vitreous_3"


def test_by_tissue_all_animals(lens2, lens3):
    """check tissue number csv file produced
    in by_tissue_all_animals directory
    containing data from all included animals"""
    lens2.do_all_the_things()
    lens3.do_all_the_things()
    by.by_tissue_all_animals("lens")
    path = Path('../../data/banbury/_data/by_tissue_all_animals/lens.csv')
    assert Path(path).exists() == True
    query = np.genfromtxt(path, dtype='U', delimiter=',', usecols=0)
    assert query[0] == "animal_2_lens_0"
    assert query[91] == "animal_3_lens_3"


def test_all_data(lens2, lens3, vitreous2):
    """check all data csv file produced
    in _data directory
    containing data from all included animals and tissues"""
    lens2.do_all_the_things()
    lens3.do_all_the_things()
    vitreous2.do_all_the_things()
    by.all_data()
    path = Path('../../data/banbury/_data/all_data.csv')
    assert Path(path).exists() == True
    query = np.genfromtxt(path, dtype='U', delimiter=',', usecols=0)
    assert query[0] == "animal_2_lens_0"
    assert query[95] == "animal_2_vitreous_7"
    assert query[179] == "animal_3_lens_3"
