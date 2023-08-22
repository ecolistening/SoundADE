import numpy as np


def dc_offset(audio: np.ndarray) -> np.ndarray:
    return audio - np.mean(audio)
