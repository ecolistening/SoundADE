import pytest
import numpy as np

from soundade.audio.binarisation import threshold

def test_positive_values():
    input_data = np.array([1, 2, 3, 4, 5])
    expected_output = np.array([True, True, True, True, True])
    assert (threshold(input_data) == expected_output).all()

def test_zero_values():
    input_data = np.array([0, 0, 0, 0, 0])
    expected_output = np.array([False, False, False, False, False])
    assert (threshold(input_data) == expected_output).all()

def test_negative_values():
    input_data = np.array([-1, -2, -3, -4, -5])
    expected_output = np.array([False, False, False, False, False])
    assert (threshold(input_data) == expected_output).all()

def test_mixed_values():
    input_data = np.array([-1, 0, 2, -3, 4])
    expected_output = np.array([False, False, True, False, True])
    assert (threshold(input_data) == expected_output).all()
