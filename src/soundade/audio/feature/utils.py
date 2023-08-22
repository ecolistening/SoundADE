import numpy as np


def pad(y, frame_length, mode='constant'):
    '''Pad audio for the computation of windowed AI features that do not automatically pad data.

    @TODO Write doctests

    :param y:
    :param frame_length:
    :param mode:
    :return:
    '''
    padding = [(0, 0) for _ in range(y.ndim)]
    padding[-1] = (int(frame_length // 2), int(frame_length // 2))
    return  np.pad(y, padding, mode=mode)


def window(y, frame_length, hop_length):
    '''Windowize an audio file for a given frame and hop length.

    @TODO Write doctests

    :param y:
    :param frame_length:
    :param hop_length:
    :return:
    '''
    ws = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length,:]
    return [w.flatten() for w in np.split(ws, ws.shape[0], axis=0)]


def pad_window(y, frame_length, hop_length, mode='constant'):
    return window(pad(y, frame_length, mode), frame_length, hop_length)
