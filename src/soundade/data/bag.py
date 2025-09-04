import itertools
import logging
import librosa
import maad
import maad.features
import maad.sound
import numpy as np
import scipy
import soundfile
import struct
import uuid
import os

from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Iterable, Callable, Tuple

from soundade.audio.feature.scalar import Features as ScalarFeatures
from soundade.audio.feature.vector import Features, do_spectrogram
from soundade.audio.filter import dc_offset

FRAME_LENGTH, HOP_LENGTH = 16000, 4000

logging.basicConfig(level=logging.INFO)

def file_to_cid(file_path):
    import multiformats
    with open(file_path, 'rb') as f:
        data = f.read()
    digest = multiformats.multihash.digest(data, "sha2-256")
    return multiformats.CID("base32", 1, "raw", digest)

def file_segment_to_cid(file_path, seconds=60):
    import multiformats
    with soundfile.SoundFile(file_path) as f:
        sample_rate = f.samplerate
        channels = f.channels
        frames_to_read = int(sample_rate * seconds)
        frames = f.read(frames_to_read, dtype='int16', always_2d=True)
        raw_bytes = frames.tobytes()
        digest = multihash.digest(raw_bytes, "sha2-256")
        cid = CID("base32", 1, "raw", digest)
        return str(cid)

INVALID_AUDIO_DICT = {
    "valid": False,
    "duration": None,
    "sr": None,
    "channels": None,
}

class MPEGHeaderError(Exception):
    pass

def validate_MPEG_header(file_path):
    with open(file_path, 'rb') as f:
        header_bytes = f.read(4)
        if len(header_bytes) < 4:
            raise MPEGHeaderError("Illegal Audio-MPEG Header, invalidating file")
        header_int = struct.unpack('>H', header_bytes[:2])[0]
        if (header_int >> 4) & 0xFFF == 0xFFF or (header_int >> 5) & 0x7FF == 0x7FF:
            pass
        else:
            raise MPEGHeaderError(f"Illegal Audio-MPEG Header for {file_path}")

def valid_audio_file(file_path: str | Path):
    """
    Filter function to remove audio files that soundfile cannot parse
    """
    p = Path(file_path)
    try:
        audio_metadata = soundfile.info(p)
        return {
            "valid": True,
            "duration": audio_metadata.duration,
            "sr": audio_metadata.samplerate,
            "channels": audio_metadata.channels,
        }
    except soundfile.LibsndfileError as e:
        logging.warning(e)
        return INVALID_AUDIO_DICT


def file_path_to_audio_dict(file_path: str | Path) -> Dict[str, Any]:
    """
    Map function to transform a file path to a file index dictionary record
    """
    file_size = os.path.getsize(file_path)
    if file_size <= 0:
        return {
            "file_id": str(uuid.uuid4()),
            "file_name": Path(file_path).name,
            "local_file_path": str(file_path),
            "size": file_size,
            **INVALID_AUDIO_DICT,
        }
    audio_dict = {
        "file_id": str(uuid.uuid4()),
        "file_name": Path(file_path).name,
        "local_file_path": str(file_path),
        "size": file_size,
        **valid_audio_file(file_path),
    }
    return audio_dict

def copy_except_audio(d: Dict):
    return dict([k, d[k]] for k in set(list(d.keys())) - {'audio'})

def create_file_load_dictionary(
    audio_dict: Dict[str, Any],
    seconds: float | None = None,
    sr: int = None,
):
    """
    Definition for librosa to load audio files based on duration in seconds. Defaults to max duration

    :param file_path: Path to an audio file
    :param seconds: Duration of each audio segment
    :param sr: Leaving sr as the default None ensures that librosa will load the file at its native sample rate. 
    :return:
    """
    audio_segment_dicts = []
    duration = librosa.get_duration(path=audio_dict["local_file_path"])
    seconds = duration if seconds == -1 else seconds
    seconds = seconds or duration
    segments = int(duration // seconds)
    for i in range(segments):
        segment_dict = audio_dict.copy()
        segment_dict.update({
            "segment_id": str(uuid.uuid4()),
            "segment_idx": i,
            "offset": i * seconds,
            "duration": seconds,
            "file_duration": duration,
        })
        audio_segment_dicts.append(segment_dict)
    return audio_segment_dicts


def write_wav(d, outpath, filename_params: List = None):
    p = Path(outpath) / d['file_name']

    if filename_params is not None:
        p = p.with_stem(f'{p.stem}_{"_".join([d[fnp] for fnp in filename_params])}')

    logging.info(f'Saving file {d["local_file_path"]} to {p}')

    soundfile.write(p, d['audio'], d['sr'], subtype='FLOAT')


def remove_dc_offset(audio_dict: Dict):
    audio_dict['audio'] = dc_offset(audio_dict['audio'])
    return audio_dict


def high_pass_filter(audio_dict: Dict, fcut=300, forder=2, fname='butter', ftype='highpass'):
    audio = audio_dict['audio']
    sr = audio_dict['sr']

    try:
        audio_dict['audio'] = maad.sound.select_bandwidth(audio, sr, fcut=fcut, forder=forder, fname=fname, ftype=ftype)
    except:
        logging.error(audio_dict)
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

def load_audio_from_path(audio_dict: Dict, sr: int | None = None) -> Dict:
    '''Load audio from a path and place into a dictionary for Bag storage

    :param p: Path of file to load
    :return: Dict representation of audio file.
    '''
    sr = sr or audio_dict.get("sr")
    try:
        audio, _ = librosa.load(**{
            "path": audio_dict.get("local_file_path"),
            "sr": sr,
            "mono": True,
            "offset": audio_dict.get("offset"),
            "duration": audio_dict.get("duration"),
        })
        return {
            "file_id": audio_dict.get("file_id"),
            "path": audio_dict.get("local_file_path"),
            "segment_id": audio_dict.get("segment_id"),
            "segment_idx": audio_dict.get("segment_idx"),
            "offset": audio_dict.get("offset"),
            "duration": audio_dict.get("duration"),
            "file_duration": audio_dict.get("file_duration"),
            "sr": sr,
            "audio": audio,
        }
        return audio_dict
    except EOFError as e:
        logging.warning(f"Couldn't load file at {str(audio_dict['path'])} with offset {str(audio_dict['offset'])}")
        return None


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
    audio = audio_dict.pop('audio')
    spectrogram = do_spectrogram(
        audio,
        frame_length=frame_length,
        hop_length=hop_length
    )

    # Update
    data_dict.update({
        'frame_length': frame_length,
        'hop_length': hop_length,
        'n_fft': n_fft,
        'feature_length': 0,
    })

    if lim_from_dict:
        kwargs = kwargs | {'flim', (data_dict['low'], data_dict['high'])}

    for feature in Features:
        data_dict[feature.name] = feature.compute(
            audio,
            frame_length=frame_length,
            hop_length=hop_length,
            n_fft=n_fft,
            sr=audio_dict.get('sr'),
            spectrograms=spectrogram,
            **kwargs
        ).flatten().tolist()
        data_dict['feature_length'] = max(len(data_dict[feature.name]), data_dict['feature_length'])

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
        'frame_length': frame_length,
        'hop_length': hop_length,
        'n_fft': n_fft,
        'feature_length': 0,
    })

    if lim_from_dict:
        kwargs = kwargs | {'flim', (data_dict['low'], data_dict['high'])}

    for feature in ScalarFeatures:
        comp = feature.compute(audio, frame_length=frame_length, hop_length=hop_length, n_fft=n_fft,
                               sr=audio_dict.get('sr'), **kwargs)
        data_dict[feature.name] = comp

    return data_dict


def log_features(features_dict: Dict, features: Iterable = [], epsilon: float = 1e-8):
    for f in features:
        features_dict.update({f'log {f}': np.log(np.array(features_dict[f]) + epsilon).tolist()})
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

