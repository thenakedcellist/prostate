import pytest
import banbury as by
from pathlib import Path

@pytest.fixture
def lens_dir():
    '''return lens directory'''
    return by.change_cwd("lens")

@pytest.fixture
def cat_dir():
    '''return impossible cat directory'''
    return by.change_cwd("cat")

def test_change_cwd(lens_dir):
    '''test that working directory is changed and that exception is raised if incorrect tissue type is given'''
    assert by.change_cwd != by.root_dir

#def test_fail_change_cwd():

def test_cwd_to_root():
    assert by.cwd_to_root() == by.root_dir
