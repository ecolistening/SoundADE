import pytest
import numpy as np
import pandas as pd
import itertools
import librosa
import warnings

from numpy.typing import NDArray
from typing import Any, Dict, List

from soundade.data.bag import (
    valid_audio_file,
    file_path_to_audio_dict,
    copy_except_audio,
    create_file_load_dictionary,
    load_audio_from_path,
    remove_dc_offset,
    extract_scalar_features_from_audio,
    log_features,
    transform_features,
)
from soundade.audio.feature.scalar import Features as ScalarFeatures

@pytest.fixture(scope="session")
def wavs(file_paths, audio_params) -> List[NDArray]:
    return [librosa.load(file_path, sr=audio_params["sr"])[0] for file_path in file_paths]

def test_valid_audio_file(fixtures_path, file_paths):
    bad_file_path = fixtures_path / "audio" / "fake.wav"
    with open(bad_file_path, "w+") as f:
        f.write("Not an audio file")

    result = valid_audio_file(file_paths[0])
    assert result.get("valid") == True
    assert result.get("duration") == 60

    assert result.get("sr") == 48_000
    assert result.get("channels") == 1

    result = valid_audio_file(bad_file_path)
    assert result.get("valid") == False
    assert result.get("duration") == None
    assert result.get("sr") == None
    assert result.get("channels") == None

    bad_file_path.unlink()
    assert not bad_file_path.exists()

def test_file_path_to_audio_dict(file_paths):
    audio_dict = file_path_to_audio_dict(file_paths[0].name, root_dir=file_paths[0].parent)
    assert audio_dict.get("file_id") is not None, "'file_id' not returned"
    assert audio_dict.get("file_name") is not None, "'file_name' not returned"
    assert audio_dict.get("file_path") is not None, "'file_path' not returned"
    assert audio_dict.get("size") is not None, "'size' not returned"
    assert audio_dict.get("valid") is not None, "'valid' not returned"

def test_copy_except_audio():
    result = copy_except_audio({"other_column": "does matter", "audio": "doesnt matter"})
    assert result.get("audio") == None, "Failed to remove audio"
    assert result.get("other_column") == "does matter", "Failed to copy other columns"

def test_create_file_load_dictionary(file_paths, audio_params):
    durations = [5, 20, 60]
    for duration, file_path in itertools.product(durations, file_paths):
        segment_dicts = create_file_load_dictionary(
            {"file_path": file_path.name},
            root_dir=file_path.parent,
            seconds=duration,
        )
        assert len(segment_dicts) == int(librosa.get_duration(path=file_path) / duration), "Incorrect number of audio segments"
        for i, segment_dict in enumerate(segment_dicts):
            assert segment_dict.get("offset") == duration * i, "Audio offset is incorrect"
            assert segment_dict.get("duration") == duration, "Audio duration is incorrect"

def test_remove_dc_offset(wavs):
    wav = wavs[0]
    result = remove_dc_offset({"audio": wav})
    np.testing.assert_array_equal(result.get("audio"), wav - wav.mean(), err_msg="DC offset incorrect")

def test_load_audio_from_path(file_paths, audio_params):
    srs = [8000, 24000, 48000]
    durations = [5, 20, 60]
    for sr, duration, file_path in itertools.product(srs, durations, file_paths):
        audio_dict = load_audio_from_path(
            {"file_path": file_path.name, "offset": 0, "duration": duration},
            root_dir=file_path.parent,
            sr=sr,
        )
        assert "offset" in audio_dict, "Audio offset was not returned"
        assert "duration" in audio_dict, "Audio duration was not returned"
        assert type(audio_dict.get("audio")) == np.ndarray, "Audio was not loaded"
        assert audio_dict.get("sr") == sr, "Sample rate is missing / incorrect"
        assert audio_dict.get("audio").shape[0] == sr * duration, "Audio was not loaded at specified sample rate"

def test_extract_scalar_features_from_audio(file_paths, wavs, audio_params):
    sr = audio_params.pop("sr") # Remove to prevent duplicate argument
    actual = pd.DataFrame([
        extract_scalar_features_from_audio(
            {"file_path": file_path, "audio": audio, "sr": sr},
            **audio_params,
        )
        for file_path, audio in zip(file_paths, wavs)
    ])
    base_columns = ["file_path", "sr", "frame_length", "hop_length", "n_fft"]
    feature_columns = [feature.name for feature in ScalarFeatures]
    assert sorted(actual.columns) == sorted([*base_columns, *feature_columns]), "Columns missing"

def test_extract_vector_features_from_audio():
    warnings.warn("'extract_vector_features_from_audio' test not yet implemented", UserWarning)

def test_log_features():
    result = log_features(
        {"A": np.ones(1), "B": np.ones(1)},
        features=["A"],
    )
    assert (result.get("log A") == np.zeros(1)).all(), "Failed to log features"
    assert result.get("log B") == None, "Applied log to wrong feature"

def test_transform_features():
    result = transform_features(
        {"A": np.ones(2), "B": np.ones(2)},
        features=["A"],
        name="{f} squared",
        transformation=lambda x: x**2,
    )
    assert (result.get("A squared") == np.ones(2) ** 2).all(), "Failed to transform features"
    assert result.get("B squared") == None, "Applied transform to wrong feature"
