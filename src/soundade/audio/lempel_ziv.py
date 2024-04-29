import random
import warnings
from ctypes import ArgumentError
from typing import Callable, Dict

import numpy as np
import pandas as pd


def lempel_ziv_complexity(sequence: np.ndarray, normalisation: str = 'random', minimum: bool = False,
                          n_symbols=None) -> int:
    '''Lempel-Ziv-Welch compression of binary input sequence, e.g. sequence='0010101'. It outputs the size of the dictionary of binary words.

    Based on code from Adam Barrett with a normalization implementation derived from antropy (https://raphaelvallat.com/antropy/build/html/index.html)

    There are 3 types of normalization:
    'random' divides the lz complexity by the complexity of a random bit sequence of the same length.
    'proportional' divides the lz complexity by the complexity of a random bit sequence with the same 0-to-1 ratio
    'zhang' (from https://raphaelvallat.com/antropy/build/html/generated/antropy.lziv_complexity.html#antropy.lziv_complexity) divides the lz complexity by (n / log_2 (n)) where n is the length of the bit sequence

    :param normalisation: None or ['random','proportional','zhang']
    :return:
    '''

    # Compute the dictionary
    d = lempel_ziv_dictionary(sequence)

    # Normalise to the minimum theoretical value as well
    if minimum:
        try:
            m = minimum_normalisation(sequence)
            l = len(d) - m
            if normalisation == 'random':
                return l / (random_normalisation(sequence, n_symbols) - m)
            elif normalisation == 'proportional':
                return l / (proportional_normalisation(sequence) - m)
            elif normalisation == 'zhang':
                return l / (zhang_normalisation(sequence) - m)
            elif normalisation is None:
                return l
        except ZeroDivisionError as e:
            warnings.warn(
                f'Divided by zero: sequence length {len(sequence.tolist())}, dictionary length {len(d)}')
            return np.nan

        raise ArgumentError('Valid normalization options are "random", "proportional", "zhang" and None.')

    # Do normalisation, if necessary
    try:
        if normalisation == 'random':
            return len(d) / random_normalisation(sequence, n_symbols)
        elif normalisation == 'proportional':
            return len(d) / proportional_normalisation(sequence)
        elif normalisation == 'zhang':
            return len(d) / zhang_normalisation(sequence)
        elif normalisation is None:
            return len(d)
    except ZeroDivisionError as e:
        warnings.warn(f'Divided by zero: sequence length {len(sequence.tolist())}, dictionary length {len(d)}')
        return np.nan

def lempel_ziv_dictionary(sequence: np.ndarray):
  """Compute the dictionary of binary words using Lempel-Ziv algorithm.

  Args:
    sequence (np.ndarray): The input binary sequence.

  Returns:
    set: The dictionary of binary words.

  Examples:
    >>> sequence = np.array([0, 0, 1, 0, 1, 0, 1])
    >>> lzd = lempel_ziv_dictionary(sequence)
    >>> sorted(list(dict.fromkeys(lzd)))
    [(0,), (0, 1), (0, 1, 0), (1,)]

    >>> sequence = np.array([1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0])
    >>> lzd = lempel_ziv_dictionary(sequence)
    >>> sorted(list(dict.fromkeys(lzd)))
    [(0,), (0, 0), (0, 0, 0), (0, 1), (0, 1, 0), (0, 1, 0, 1), (1,), (1, 0), (1, 1), (1, 1, 0)]
  """

  d = set([])
  w = []
  for c in sequence.tolist():
    wc = w + [c]
    if tuple(wc) in d:
      w = wc
    else:
      d.add(tuple(wc))
      w = []
      # In previous versions, this was written as w = [c] so that the next word contains the last letter
      # of this word. There is no good reason to do this and it has been removed.

  return d


def random_normalisation(s: np.ndarray, n_symbols=None) -> float:
    try:
        n = s.size
    except AttributeError as e:
        n = len(s)

    if n_symbols is None:
        t = np.random.random(size=n) > 0.5
    else:
        t = np.random.randint(0, n_symbols, size=n)

    return lempel_ziv_complexity(t, normalisation=None)


def proportional_normalisation(s: np.ndarray) -> float:
    try:
        t = s.copy()
        np.random.shuffle(t)
    except AttributeError as e:
        l = list(s)
        random.shuffle(l)
        t = ''.join(l)

    return lempel_ziv_complexity(t, normalisation=None)


def zhang_normalisation(s: np.ndarray) -> float:
    try:
        n = s.size
    except AttributeError as e:
        n = len(s)
    return (n / np.log2(n))


def minimum_normalisation(s: np.ndarray) -> float:
    try:
        n = s.size
    except AttributeError as e:
        n = len(s)

    t = np.ones(n)

    return lempel_ziv_complexity(t, normalisation=None)


def lzc_row(row, binarisation: Callable, complexity: Callable) -> pd.Series:
    '''Compute the lempel-ziv complexity of a dataframe row

    :param row: The row to compute complexity on. Should include only the columns containing relevant data
    :param binarisations: dict of C
    :return:
    '''
    r = row.to_numpy().astype(float)  # Convert to numpy float array
    r = r[~np.isnan(r)].flatten()  # Remove NANs
    # Do this beforehand.
    # r = r[pad_frames:-pad_frames]  # Remove frames affected by padding

    # Binarisations
    # lz = dict([(b, lempel_ziv_complexity(binarisations[b](r), normalisation='proportional')) for b in binarisations])

    # lz = dict([(b, complexity(binarisations[b](r))) for b in binarisations])
    #
    # return pd.Series(lz, name=row.name)
    try:
        return complexity(binarisation(r))
    except ValueError as e:
        warnings.warn(str(e))
        return np.nan


def lzcs_row(row, binarisations: Dict[str, Callable], complexity: Callable) -> pd.Series:
    '''Compute the lempel-ziv complexity of a dataframe row

    :param row: The row to compute complexity on. Should include only the columns containing relevant data
    :param binarisations: dict of C
    :return:
    '''
    r = row.to_numpy().astype(float)  # Convert to numpy float array
    r = r[~np.isnan(r)].flatten()  # Remove NANs
    # Do this beforehand.
    # r = r[pad_frames:-pad_frames]  # Remove frames affected by padding

    # Binarisations
    # lz = dict([(b, lempel_ziv_complexity(binarisations[b](r), normalisation='proportional')) for b in binarisations])

    # lz = dict([(b, complexity(binarisations[b](r))) for b in binarisations])
    #
    # return pd.Series(lz, name=row.name)
    d = {}
    for b in binarisations:
        try:
            d[b] = complexity(binarisations[b](r))
        except ValueError as e:
            warnings.warn(str(e))
            d[b] = np.nan
    return d


def windowed_lzc(dataframe, window, hop, fps, complexity, binarisation):
    '''Compute the lempel-ziv complexity in the windowed sections of a dataframe row
    
    :param dataframe: 
    :param window: 
    :param hop: 
    :param fps: 
    :param complexity: 
    :param binarisation: 
    :return: 
    '''
    metadata = dataframe.iloc[:dataframe.index.get_loc('0')]
    features = dataframe.iloc[dataframe.index.get_loc('0'):]

    windowed = pd.DataFrame(
        np.lib.stride_tricks.sliding_window_view(features, window_shape=window, axis=0).reshape(-1, window)[::hop, :])
    lz = windowed.apply(lzc_row, axis=1, binarisation=binarisation, complexity=complexity)
    lz.index = lz.index * hop / fps
    return pd.concat([metadata, lz])
