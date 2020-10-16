import pytest
from banbury import ByTissueByAnimal
from pathlib import Path

#tissue_type = ["cornea", "lens", "optic_nerve", "retina", "vitreous"]
#animal_number = (range(1, 12))

@pytest.fixture
def lens2():
    """returns ByTissueByAnimal instance of lens 2"""
    return ByTissueByAnimal("lens", 2)

def test_file_search(lens2):
    """check that directory for input tissue and animal number exists, and that a non-empty dictionary is created"""
    lens2.file_search()
    assert lens2.file_dir.exists() == True
    assert len(lens2.file_dict) != 0

def test_x_data_extraction(lens2):
    """check that x data are extracted from first file in directory, and that an array with data in rows is created"""
    lens2.file_search()
    lens2.x_data_extraction()
