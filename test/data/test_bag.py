import pytest

import numpy as np
import pandas as pd
import librosa

from numpy.typing import NDArray
from typing import Any, Dict, List

from soundade.data.bag import (
    create_file_load_dictionary,
    load_audio_from_path,
    extract_scalar_features_from_audio,
)
from soundade.audio.feature.scalar import Features as ScalarFeatures

@pytest.fixture
def wavs(file_paths, audio_params) -> List[NDArray]:
    return [librosa.load(file_path, sr=audio_params["sr"])[0] for file_path in file_paths]

def test_create_file_load_dictionary(file_paths, audio_params):
    seconds = 20
    sr = audio_params["sr"]
    for file_path in file_paths:
        segment_dicts = create_file_load_dictionary(
            {"file_path": file_path.name},
            root_dir=file_path.parent,
            seconds=seconds,
            sr=sr,
        )
        assert len(segment_dicts) == int(librosa.get_duration(path=file_path, sr=sr) / seconds)
        for i, segment_dict in enumerate(segment_dicts):
            assert segment_dict.get("offset") == seconds * i
            assert segment_dict.get("duration") == seconds

def test_load_audio_from_path(file_paths, audio_params):
    sr = audio_params["sr"]
    for file_path in file_paths:
        audio_dict = load_audio_from_path(
            {"file_path": file_path.name, "offset": 0, "seconds": 60},
            root_dir=file_path.parent,
            sr=sr,
        )
        assert "offset" in audio_dict
        assert "duration" in audio_dict
        assert type(audio_dict.get("audio")) == np.ndarray

def test_extract_scalar_features_from_audio(file_paths, wavs, audio_params):
    sr = audio_params.pop("sr")
    actual = pd.DataFrame([
        extract_scalar_features_from_audio(
            {"file_path": file_path, "audio": audio, "sr": sr},
            **audio_params,
        )
        for file_path, audio in zip(file_paths, wavs)
    ])
    base_columns = ["file_path", "sr", "frame_length", "hop_length", "n_fft"]
    feature_columns = [feature.name for feature in ScalarFeatures]
    assert sorted(actual.columns) == sorted([*base_columns, *feature_columns])
