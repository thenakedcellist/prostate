import pytest
import banbury as by
from pathlib import Path


@pytest.mark.parametrize('tissue', [
    "cornea",
    "lens",
    "optic nerve",
    "retina",
    "vitreous",
])

def test_change_cwd(tissue):
    by.change_cwd(tissue)
