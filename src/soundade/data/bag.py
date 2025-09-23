import itertools
import logging
import librosa
import maad
import maad.features
import maad.sound
import numpy as np
import os
import soundfile
import uuid

from pathlib import Path
from typing import Any, Dict, List, Iterable, Callable, Tuple

from soundade.audio.feature.scalar import Features as ScalarFeatures
from soundade.audio.feature.vector import Features, do_spectrogram
from soundade.audio.filter import dc_offset

FRAME_LENGTH, HOP_LENGTH = 16000, 4000

logging.basicConfig(level=logging.INFO)

INVALID_AUDIO_DICT = {
    "valid": False,
    "duration": None, "sr": None,
    "channels": None,
}

def valid_audio_file(file_path: str | Path):
    """Filter function to mark audio files that soundfile cannot parse as invalid

    :param file_path str | pathlib.Path: path to the audio file
    :return a dictionary of the form { valid: bool, duration: float, sr: int, channels: int }
    """
    try:
        audio_metadata = soundfile.info(Path(file_path))
        return {
            "valid": True,
            "duration": audio_metadata.duration,
            "sr": audio_metadata.samplerate,
            "channels": audio_metadata.channels,
        }
    except soundfile.LibsndfileError as e:
        logging.warning(e)
        return INVALID_AUDIO_DICT

def file_path_to_audio_dict(file_path: str | Path, root_dir: str | Path) -> Dict[str, Any]:
    """Transform a file path to a file index dictionary record

    :param file_path str | pathlib.Path: relative path to the audio file
    :param root_dir str | pathlib.Path: root directory of the audio
    :return a dictionary of the form:
            { file_id: str, file_name: str, file_path: str, size: int, valid: bool, duration: float, sr: int, channels: int }
    """
    file_size = os.path.getsize(root_dir / file_path)
    return {
        "file_id": str(uuid.uuid4()),
        "file_name": Path(file_path).name,
        "file_path": str(file_path),
        "size": file_size,
        **valid_audio_file(root_dir / file_path),
    }

def copy_except_audio(d: Dict):
    return dict([k, d[k]] for k in set(list(d.keys())) - {'audio'})

def create_file_load_dictionary(
    audio_dict: Dict[str, Any],
    root_dir: Path,
    seconds: float | None = None,
    sr: int | None = None,
) -> List[Dict[str, Any]]:
    """Create N offset dictonaries for each S length audio segment

    :param audio_dict: A dictionary of the form:
           { file_id: str, segment_id: str | int, file_path: str }
    :param seconds: Duration of each audio segment
    :param sr: Leaving sr as the default None ensures that librosa will load the file at its native sample rate. 
    :return:
    """
    audio_segment_dicts = []
    duration = librosa.get_duration(
        path=root_dir / audio_dict["file_path"],
        sr=sr or audio_dict.get("sr")
    )
    seconds = duration if seconds == -1 else seconds
    seconds = seconds or duration
    segments = int(duration // seconds)
    for t in range(segments):
        segment_dict = audio_dict.copy()
        segment_dict.update({
            "segment_id": str(uuid.uuid4()),
            "segment_idx": t,
            "offset": t * seconds,
            "duration": seconds,
        })
        audio_segment_dicts.append(segment_dict)
    return audio_segment_dicts

def remove_dc_offset(audio_dict: Dict):
    audio_dict['audio'] = dc_offset(audio_dict['audio'])
    return audio_dict

def apply_high_pass_filter(audio_dict: Dict, fcut=300, forder=2, fname='butter', ftype='highpass'):
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

def load_audio_from_path(audio_dict: Dict, root_dir: Path, sr: int | None = None) -> Dict:
    """Load audio from a path and place into a dictionary for bag storage

    :param audio_dict: A dictionary of the form:
           { file_id: str | int, segment_id: str | int, file_path: str, offset: float, duration: float }
    :param root_dir: Root path of audio
    :return: Dict representation of audio file.
    """
    sr = sr or audio_dict.get("sr")
    try:
        audio, _ = librosa.load(**{
            "path": root_dir / audio_dict.get("file_path"),
            "sr": sr,
            "mono": True,
            "offset": audio_dict.get("offset"),
            "duration": audio_dict.get("duration"),
        })
        return {
            "file_id": audio_dict.get("file_id"),
            "segment_id": audio_dict.get("segment_id"),
            "segment_idx": audio_dict.get("segment_idx"),
            "offset": audio_dict.get("offset"),
            "duration": audio_dict.get("duration"),
            "sr": sr,
            "audio": audio,
        }
        return audio_dict
    except EOFError as e:
        logging.warning(f"Couldn't load file at {str(root_dir / audio_dict['file_path'])} with offset {str(audio_dict['offset'])}")
        return None

def extract_vector_features_from_audio(
    audio_dict: Dict,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
    n_fft: int = FRAME_LENGTH,
    lim_from_dict: bool = False,
    **kwargs: Any,
) -> Dict:
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

def extract_scalar_features_from_audio(
    audio_dict: Dict,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
    n_fft: int = FRAME_LENGTH,
    lim_from_dict: bool = False,
    **kwargs: Any,
) -> Dict:
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
