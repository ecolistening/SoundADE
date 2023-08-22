import unittest

import numpy as np

from soundade.audio.binarisation import mean_threshold


class TestMeanThresholdFunction(unittest.TestCase):

    def test_positive_values(self):
        input_data = np.array([1, 2, 3, 4, 5])
        expected_output = np.array([False, False, False, True, True])
        self.assertTrue(np.array_equal(mean_threshold(input_data), expected_output))

    def test_zero_values(self):
        input_data = np.array([0, 0, 0, 0, 0])
        expected_output = np.array([False, False, False, False, False])
        self.assertTrue(np.array_equal(mean_threshold(input_data), expected_output))

    def test_negative_values(self):
        input_data = np.array([-1, -2, -3, -4, -5])
        expected_output = np.array([True, True, False, False, False])
        self.assertTrue(np.array_equal(mean_threshold(input_data), expected_output))

    def test_mixed_values(self):
        input_data = np.array([-1, 0, 2, -3, 4])
        expected_output = np.array([False, False, True, False, True])
        self.assertTrue(np.array_equal(mean_threshold(input_data), expected_output))


if __name__ == '__main__':
    unittest.main()
