import pytest

import librosa
import os
import numpy as np
import pandas as pd
import pathlib
import subprocess
import yaml

from numpy.typing import NDArray
from scipy.stats import pearsonr
from typing import Any, List, Dict

from soundade.audio.feature import scalar

# @pytest.fixture(scope="session", autouse=True)
# def extract_features():
#     subprocess.run(["Rscript", "test/helpers/indices.R", "test/fixtures/audio", "test/fixtures/audio_params.yaml", "/test/fixtures/indices.parquet"], check=True)
#     yield
#     pathlib.Path("test/fixtures/indices.parquet").unlink()

@pytest.fixture(scope="session")
def wavs(file_paths, audio_params) -> List[NDArray]:
    return [librosa.load(file_path, sr=audio_params["sr"])[0] for file_path in file_paths]

@pytest.fixture(scope="session")
def spectrograms(wavs, audio_params) -> List[NDArray]:
    return [scalar.spectrogram(wav, **audio_params) for wav in wavs]

@pytest.fixture(scope="session")
def expected_acoustic_features(fixtures_path) -> pd.DataFrame:
    df = pd.read_parquet(fixtures_path / "indices.parquet")
    df = df.set_index("file_name").sort_index()
    df.index.name = None
    return df.astype({col: np.float32 for col in df.columns})

def test_acoustic_complexity_index(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    expected_acoustic_features: List[pd.DataFrame]
) -> None:
    metric = "acoustic_complexity_index"
    fn = getattr(scalar, metric)
    expected = expected_acoustic_features[metric].to_frame()
    results = [fn(y=wav, **audio_params) for wav in wavs]
    actual = pd.DataFrame(data=results, columns=[metric], index=file_names).sort_index().astype(np.float32)
    pd.testing.assert_frame_equal(expected, actual, atol=1e-1, rtol=1e-3)

def test_acoustic_evenness_index(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    expected_acoustic_features: List[pd.DataFrame]
) -> None:
    metric = "acoustic_evenness_index"
    fn = getattr(scalar, metric)
    expected = expected_acoustic_features[metric].to_frame()
    results = [fn(y=wav, **audio_params, R_compatible=True) for wav in wavs]
    actual = pd.DataFrame(data=results, columns=[metric], index=file_names).sort_index().astype(np.float32)
    pd.testing.assert_frame_equal(expected, actual, atol=1e-1, rtol=1e-8)

def test_bioacoustic_index(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    expected_acoustic_features: List[pd.DataFrame]
) -> None:
    metric = "bioacoustic_index"
    fn = getattr(scalar, metric)
    expected = expected_acoustic_features[metric].to_frame()
    results = [fn(y=wav, **audio_params, R_compatible=True) for wav in wavs]
    actual = pd.DataFrame(data=results, columns=[metric], index=file_names).sort_index().astype(np.float32)
    pd.testing.assert_frame_equal(expected, actual, atol=1e-1, rtol=1e-3)

def test_spectral_flux(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    expected_acoustic_features: List[pd.DataFrame]
) -> None:
    metric = "spectral_flux"
    fn = getattr(scalar, metric)
    expected = expected_acoustic_features[metric].to_frame()
    results = [fn(y=wav, **audio_params) for wav in wavs]
    actual = pd.DataFrame(data=results, columns=[metric], index=file_names).sort_index().astype(np.float32)
    pd.testing.assert_frame_equal(expected, actual, atol=1e-3, rtol=1e-3)

def test_zero_crossing_rate(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    expected_acoustic_features: List[pd.DataFrame]
) -> None:
    metric = "zero_crossing_rate"
    fn = getattr(scalar, metric)
    expected = expected_acoustic_features[metric].to_frame()
    results = [fn(y=wav, frame_length=audio_params["frame_length"], hop_length=audio_params["hop_length"]) for wav in wavs]
    actual = pd.DataFrame(data=results, columns=[metric], index=file_names).sort_index().astype(np.float32)
    pd.testing.assert_frame_equal(expected, actual, atol=1e-3, rtol=1e-3)

def test_spectral_centroid(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    expected_acoustic_features: List[pd.DataFrame]
) -> None:
    metric = "spectral_centroid"
    fn = getattr(scalar, metric)
    expected = expected_acoustic_features[metric].to_frame()
    results = [fn(y=wav, **audio_params) for wav in wavs]
    actual = pd.DataFrame(data=results, columns=[metric], index=file_names).sort_index().astype(np.float32)
    pd.testing.assert_frame_equal(expected, actual, atol=1e3, rtol=1e-3) # :-S

def test_root_mean_square(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    expected_acoustic_features: List[pd.DataFrame]
) -> None:
    metric = "root_mean_square"
    fn = getattr(scalar, metric)
    expected = expected_acoustic_features[metric].to_frame()
    results = [fn(y=wav, **audio_params) for wav in wavs]
    actual = pd.DataFrame(data=results, columns=[metric], index=file_names).sort_index().astype(np.float32)
    pd.testing.assert_frame_equal(expected, actual, atol=1e-3, rtol=1e-3)

def test_spectral_entropy(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    expected_acoustic_features: List[pd.DataFrame]
) -> None:
    metric = "spectral_entropy"
    fn = getattr(scalar, metric)
    expected = expected_acoustic_features[metric].to_frame()
    results = [fn(y=wav, **audio_params) for wav in wavs]
    actual = pd.DataFrame(data=results, columns=[metric], index=file_names).sort_index().astype(np.float32)
    pd.testing.assert_frame_equal(expected, actual, atol=1e-3, rtol=1e-3)

def test_temporal_entropy(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    expected_acoustic_features: List[pd.DataFrame]
) -> None:
    metric = "temporal_entropy"
    fn = getattr(scalar, metric)
    expected = expected_acoustic_features[metric].to_frame()
    results = [fn(y=wav, **audio_params) for wav in wavs]
    actual = pd.DataFrame(data=results, columns=[metric], index=file_names).sort_index().astype(np.float32)
    pd.testing.assert_frame_equal(expected, actual, atol=1e-3, rtol=1e-3)
