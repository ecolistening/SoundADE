from typing import List

import numpy as np
def symbolise(x: np.ndarray, bins: List) -> np.ndarray:
    return np.digitize(x, bins=bins)

def bin_edges(x: np.ndarray, n_bins: int) -> List:
    return np.histogram_bin_edges(x)