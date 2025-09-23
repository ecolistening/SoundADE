from typing import Sequence, Union

import numpy as np
from sklearn.cluster import k_means


def threshold(x: np.ndarray) -> np.ndarray:
    '''Binarises a sequence based on thresholding.

    :param x: Sequence to binarise
    :return: Binary sequence
    '''
    return x > 0


def mean_threshold(x: np.ndarray, window=None) -> np.ndarray:
    '''Binarises a sequence based on thresholding values above and below the mean.

    :param x: Sequence to binarise
    :param window: Width of a rolling window used to calculate the mean instead of the universal mean. Default: None
    :return: Binary sequence
    '''
    if window is not None:
        windowed_mean = np.lib.stride_tricks.sliding_window_view(x, window).mean(axis=1)
        return x[window // 2:(windowed_mean.size + window // 2)] > windowed_mean

    return x > np.mean(x)


def median_threshold(x: np.ndarray, window=None) -> np.ndarray:
    '''Binarises a sequence based on thresholding values above and below the median.

    :param x: Sequence to binarise
    :param window: Width of a rolling window used to calculate the median instead of the universal median. Default: None
    :return: Binary sequence
    '''
    if window is not None:
        windowed_median = np.median(np.lib.stride_tricks.sliding_window_view(x, window), axis=1)
        return x[window // 2:(windowed_median.size + window // 2)] > windowed_median

    return x > np.median(x)


def difference(x: np.ndarray, **kwargs) -> np.ndarray:
    """Binarises a sequence based trajectory, or the difference between two values.
    If the difference is 0, the binarisation returns 0.

    :param x: Sequence to binarise
    :return: Binary sequence
    """
    return np.diff(x, **kwargs) > 0


def pdm(x: np.ndarray) -> np.ndarray:  # , np.array):
    """
    Performs Pulse Density Modulation on the audio stream and returns the PDM representation and error

    Based on code from https://gist.github.com/jeanminet/2913ca7a87e96296b27e802575ad6153

    :return:
    """

    '''Code from Chris
    def pdm(x, oversample=1):
     n = len(x)
     y = np.zeros(n * oversample)
     idx=0
     error = np.zeros(len(y) + 1)
     for i in range(n):
         for j in range(oversample):
             y[idx] = 1 if x[i] >= error[idx] else 0
             error[idx+1] = y[idx] - x[i] + error[idx]
             idx += 1
     return y, error[0:n]
    '''

    n = x.size
    y = np.zeros(n)
    error = np.zeros(n + 1)
    qe = 0.0

    for i in range(n):
        qe += x[i]
        y[i] = 1 if qe > 0 else -1
        qe -= y[i]

    # for i in range(n):
    #     y[i] = 1 if x[i] >= error[i] else 0
    #     error[i + 1] = y[i] - x[i] + error[i]
    y[y < 0] = 0

    return y  # , error[0:n]


def kmeans(x: Union[np.ndarray, Sequence[np.ndarray]], use_time=False, **kwargs):
    '''

    :param x: (N,)-element numpy array containing audio feature data or a sequence of (N,)-element numpy arrays.
    :param use_time: Use the timing of the data points as a feature.
    :param kwargs:
    :return:
    '''
    try:
        X = x.reshape(-1, 1)
    except AttributeError:
        X = np.concatenate([a.reshape(-1, 1) for a in x], axis=1)

    if use_time:
        X = np.concatenate([X, np.linspace(0, 1.0, X.shape[1]).reshape(-1, 1)], axis=1)

    centroid, label, inertia = k_means(X, 2)

    return label
