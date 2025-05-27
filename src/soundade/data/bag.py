import itertools
import logging
from datetime import timedelta
from typing import Dict, List, Iterable, Callable, Tuple

import librosa
import maad
import maad.features
import maad.sound
import numpy as np
import scipy

from soundade.audio.feature.scalar import Features as ScalarFeatures
from soundade.audio.feature.vector import Features, do_spectrogram
from soundade.audio.filter import dc_offset

FRAME_LENGTH, HOP_LENGTH = 16000, 4000


def copy_except_audio(d: Dict):
    return dict([k, d[k]] for k in set(list(d.keys())) - {'audio'})


def create_file_load_dictionary(files: List, sr=None):
    """
    
    :param files: 
    :param sr: Leaving sr as the default None ensures that librosa will load the file at its native sample rate. 
    :return: 
    """
    audio_dicts = []
    for f in files:
        d = {
            'path': f,
            'sr': sr,
        }
        audio_dicts.append(d)

    return audio_dicts


def create_file_segment_load_dictionary(files: List, seconds: float = 60.0, sr=None):
    audio_segments_dicts = []
    for f in files:
        duration = librosa.get_duration(filename=f, sr=sr)
        segments = int(duration // seconds)
        for i in range(segments):
            d = {
                'path': f,
                'sr': sr,
                'offset': i * seconds,
                'duration': seconds
            }
            audio_segments_dicts.append(d)

    return audio_segments_dicts


def remove_dc_offset(audio_dict: Dict):
    audio_dict['audio'] = dc_offset(audio_dict['audio'])
    return audio_dict


def high_pass_filter(audio_dict: Dict, fcut=300, forder=2, fname='butter', ftype='highpass'):
    audio = audio_dict['audio']
    sr = audio_dict['sr']

    audio_dict['audio'] = maad.sound.select_bandwidth(audio, sr, fcut=fcut, forder=forder, fname=fname, ftype=ftype)

    return audio_dict


def extract_banded_audio(audio_dict: Dict, bands: Iterable[Tuple[int, int]]):
    audio = audio_dict['audio']
    sr = audio_dict['sr']
    audio_dicts = []

    # Filter at all bands
    for low, high in bands:
        d = copy_except_audio(audio_dict)
        ba = maad.sound.select_bandwidth(audio, fs=sr, fcut=[low, high], forder=2, ftype='bandpass')

        d['low'] = low
        d['high'] = high
        d['audio'] = ba

        audio_dicts.append(d)

    return audio_dicts


def load_audio_from_path(audio_dict: Dict) -> Dict:
    '''Load audio from a path and place into a dictionary for Bag storage

    :param p: Path of file to load
    :return: Dict representation of audio file.
    '''
    try:
        audio, sr = librosa.load(**audio_dict)

        data_dict = {
            'path': str(audio_dict['path']),
            'file': audio_dict['path'].name,
            'audio': audio,
            'sr': sr
        }

        return data_dict
    except EOFError as e:
        logging.warning(f"Couldn't load file at {str(audio_dict['path'])}.")

        return None


def load_audio_segment_from_path(audio_segment_dict: Dict) -> Dict:
    '''Load audio from a path and place into a dictionary for Bag storage

    :param p: Path of file to load
    :return: Dict representation of audio file.
    '''
    audio, sr = librosa.load(**audio_segment_dict)

    data_dict = {
        'path': str(audio_segment_dict['path']),
        'file': audio_segment_dict['path'].name,
        'audio': audio,
        'sr': sr
    }

    # Add these if they exist
    try:
        data_dict |= {'offset': audio_segment_dict['offset']}
        data_dict |= {'duration': audio_segment_dict['duration']}
    except KeyError as e:
        pass

    return data_dict


def split_audio(audio_dict: Dict, seconds: int = 60) -> List[Dict]:
    '''NOT USED. Split audio contained in an audio_dict into S-second long segments

    :param audio_dict:
    :param seconds:
    :return:
    '''
    # split_audio_dict = audio_dict.copy()

    audio: np.ndarray = audio_dict.get('audio')
    split_audio_dict = copy_except_audio(audio_dict)
    sr = split_audio_dict.get('sr')

    segment_length = int(sr * seconds)

    splits = np.split(audio,
                      range(segment_length, audio.size, segment_length))  # Split audio on multiples of segment_length
    splits = filter(lambda s: s.size == segment_length, splits)  # Remove trailing segments that are mismatched

    return [split_audio_dict | {
        'time offset': i * timedelta(seconds=seconds),
        'segment length': timedelta(seconds=seconds),
        'audio': s,
    } for i, s in enumerate(splits)]


def power_spectra_from_audio(audio_dict: Dict, **kwargs):
    # Make sure to copy ALL data from the data dict
    data_dict = copy_except_audio(audio_dict)

    # Remove the raw audio, which we don't want anymore
    audio = audio_dict.get('audio')

    data_dict['bins'], data_dict['psd'] = scipy.signal.welch(audio, fs=data_dict['sr'], nperseg=1024)

    return data_dict


def extract_features_from_audio(audio_dict: Dict, frame_length: int = FRAME_LENGTH, hop_length: int = HOP_LENGTH,
                                n_fft: int = FRAME_LENGTH, lim_from_dict=False, **kwargs) -> Dict:
    '''Extract features from audio in audio_dict

    :param lim_from_dict: Take the frequency limits (for AEI and BI) from the 'low' and 'high' entries in the dictionary
    :param audio_dict:
    :param frame_length:
    :param hop_length:
    :param n_fft:
    :param kwargs:
    :return:
    '''

    # Make sure to copy ALL data from the data dict
    data_dict = copy_except_audio(audio_dict)

    # Remove the raw audio, which we don't want anymore
    audio = audio_dict.get('audio')
    spectrogram = do_spectrogram(audio,
                                 frame_length=frame_length,
                                 hop_length=hop_length)

    # Update
    data_dict.update({
        'frame length': frame_length,
        'hop length': hop_length,
        'n fft': n_fft,
        'feature length': 0,
    })

    if lim_from_dict:
        kwargs = kwargs | {'flim', (data_dict['low'], data_dict['high'])}

    for feature in Features:
        comp = feature.compute(audio, frame_length=frame_length, hop_length=hop_length, n_fft=n_fft,
                               sr=audio_dict.get('sr'), spectrograms=spectrogram, **kwargs)
        data_dict[feature.name] = comp.flatten().tolist()
        data_dict['feature length'] = max(len(data_dict[feature.name]), data_dict['feature length'])

    return data_dict


def extract_scalar_features_from_audio(audio_dict: Dict, frame_length: int = FRAME_LENGTH, hop_length: int = HOP_LENGTH,
                                       n_fft: int = FRAME_LENGTH, lim_from_dict=False, **kwargs) -> Dict:
    '''Extract features from audio in audio_dict

    :param lim_from_dict: Take the frequency limits (for AEI and BI) from the 'low' and 'high' entries in the dictionary
    :param audio_dict:
    :param frame_length:
    :param hop_length:
    :param n_fft:
    :param kwargs:
    :return:
    '''

    # Make sure to copy ALL data from the data dict
    data_dict = copy_except_audio(audio_dict)

    # Remove the raw audio, which we don't want anymore
    audio = audio_dict.get('audio')

    # Update
    data_dict.update({
        'frame length': frame_length,
        'hop length': hop_length,
        'n fft': n_fft,
        'feature length': 0,
    })

    if lim_from_dict:
        kwargs = kwargs | {'flim', (data_dict['low'], data_dict['high'])}

    for feature in ScalarFeatures:
        comp = feature.compute(audio, frame_length=frame_length, hop_length=hop_length, n_fft=n_fft,
                               sr=audio_dict.get('sr'), **kwargs)
        data_dict[feature.name] = comp

    return data_dict


def log_features(features_dict: Dict, features: Iterable = []):
    for f in features:
        features_dict.update({f'log {f}': np.log(features_dict[f])})
    return features_dict


def transform_features(features_dict: Dict, transformation: Callable, name='{f}', features: Iterable = []):
    for f in features:
        features_dict.update({name.format(f=f): transformation(features_dict[f])})

    return features_dict


def reformat_for_dataframe(features_dict: Dict, data_keys: List = None, columns_key=None, scalar_values=False):
    # Label data columns with number from 1 .. n for data of length n
    # Default for feature processing
    if data_keys is None:
        data_keys = [f.name for f in Features]

    _features_dict = features_dict.copy()

    if columns_key is not None:
        column_names = _features_dict.pop(columns_key)

    # Set metadata
    _meta_keys = [k for k in _features_dict.keys() if k not in data_keys]
    _meta_dict = {k: _features_dict.pop(k) for k in _meta_keys}

    data_dicts = []
    for f in _features_dict:
        if scalar_values:
            data_dicts.append(_meta_dict | {'feature': f, 'value': _features_dict[f]})
        else:
            if columns_key is None:
                column_names = itertools.count()
            _labelled_columns = {f'{n}': x for n, x in zip(column_names, _features_dict[f])}

            data_dicts.append(_meta_dict | {'feature': f} | _labelled_columns)

    return data_dicts
