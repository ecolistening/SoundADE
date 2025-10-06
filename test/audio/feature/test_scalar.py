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
from typing import Any, Callable, List, Dict

from soundade.audio.feature.scalar import (
    acoustic_complexity_index,
    acoustic_evenness_index,
    bioacoustic_index,
    spectral_centroid,
    spectral_flux,
    zero_crossing_rate,
    root_mean_square,
    spectral_entropy,
    temporal_entropy,
)

# @pytest.fixture(scope="session", autouse=True)
# def extract_features():
#     subprocess.run(["Rscript", "test/helpers/indices.R", "test/fixtures/audio", "test/fixtures/audio_params.yaml", "/test/fixtures/indices.parquet"], check=True)
#     yield
#     pathlib.Path("test/fixtures/indices.parquet").unlink()

@pytest.fixture(scope="session")
def wavs(file_paths, audio_params) -> List[NDArray]:
    return [librosa.load(file_path, sr=audio_params["sr"])[0] for file_path in file_paths]

@pytest.fixture(scope="session")
def expected_acoustic_features(fixtures_path) -> pd.DataFrame:
    df = pd.read_parquet(fixtures_path / "indices.parquet")
    df = df.set_index("file_name").sort_index()
    df.index.name = None
    return df

@pytest.fixture
def expected(request, expected_acoustic_features: pd.DataFrame) -> pd.Series:
    series = expected_acoustic_features[request.param].astype(np.float32)
    series.name = None
    return series

@pytest.mark.parametrize("fn, expected", [(acoustic_complexity_index, "acoustic_complexity_index")], indirect=["expected"])
def test_acoustic_complexity_index(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    fn: Callable,
    expected: pd.Series,
) -> None:
    actual = pd.Series({file_name: fn(y=wav, **audio_params) for file_name, wav in zip(file_names, wavs)}).sort_index().astype(np.float32)
    pd.testing.assert_series_equal(expected, actual, atol=1e-1, rtol=1e-3)

@pytest.mark.parametrize("fn, expected", [(acoustic_evenness_index, "acoustic_evenness_index")], indirect=["expected"])
def test_acoustic_evenness_index(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    fn: Callable,
    expected: pd.Series,
) -> None:
    actual = pd.Series({file_name: fn(y=wav, **audio_params, R_compatible=True) for file_name, wav in zip(file_names, wavs)}).sort_index().astype(np.float32)
    pd.testing.assert_series_equal(expected, actual, atol=1e-1, rtol=1e-3)

@pytest.mark.parametrize("fn, expected", [(bioacoustic_index, "bioacoustic_index")], indirect=["expected"])
def test_bioacoustic_index(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    fn: Callable,
    expected: pd.Series,
) -> None:
    actual = pd.Series({file_name: fn(y=wav, **audio_params, R_compatible=True) for file_name, wav in zip(file_names, wavs)}).sort_index().astype(np.float32)
    pd.testing.assert_series_equal(expected, actual, atol=1e-1, rtol=1e-2)

@pytest.mark.parametrize("fn, expected", [(spectral_flux, "spectral_flux")], indirect=["expected"],)
def test_spectral_flux(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    fn: Callable,
    expected: pd.Series,
) -> None:
    actual = pd.Series({file_name: fn(y=wav, **audio_params, R_compatible=True) for file_name, wav in zip(file_names, wavs)}).sort_index().astype(np.float32)
    pd.testing.assert_series_equal(expected, actual, atol=1e-1, rtol=1e-3)

@pytest.mark.parametrize("fn, expected", [(zero_crossing_rate, "zero_crossing_rate")], indirect=["expected"])
def test_zero_crossing_rate(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    fn: Callable,
    expected: pd.Series,
) -> None:
    actual = pd.Series({file_name: fn(y=wav, **audio_params, R_compatible=True) for file_name, wav in zip(file_names, wavs)}).sort_index().astype(np.float32)
    pd.testing.assert_series_equal(expected, actual, rtol=1e-3)

@pytest.mark.parametrize("fn, expected", [(spectral_centroid, "spectral_centroid")], indirect=["expected"])
def test_spectral_centroid(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    fn: Callable,
    expected: pd.Series,
) -> None:
    actual = pd.Series({file_name: fn(y=wav, **audio_params) for file_name, wav in zip(file_names, wavs)}).sort_index().astype(np.float32)
    # slight differences in seewave / scipy spectrogram, so permit within 20% of range
    pd.testing.assert_series_equal(expected, actual, rtol=1e-1)

@pytest.mark.parametrize("fn, expected", [(root_mean_square, "root_mean_square")], indirect=["expected"])
def test_root_mean_square(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    fn: Callable,
    expected: pd.Series,
) -> None:
    actual = pd.Series({file_name: fn(y=wav, **audio_params) for file_name, wav in zip(file_names, wavs)}).sort_index().astype(np.float32)
    pd.testing.assert_series_equal(expected, actual, atol=1e-3)

@pytest.mark.parametrize("fn, expected", [(spectral_entropy, "spectral_entropy")], indirect=["expected"])
def test_spectral_entropy(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    fn: Callable,
    expected: pd.Series,
) -> None:
    actual = pd.Series({file_name: fn(y=wav, **audio_params, compatibility="seewave") for file_name, wav in zip(file_names, wavs)}).sort_index().astype(np.float32)
    pd.testing.assert_series_equal(expected, actual, atol=1e-1, rtol=1e-1)

@pytest.mark.parametrize("fn, expected", [(temporal_entropy, "temporal_entropy")], indirect=["expected"])
def test_temporal_entropy(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    fn: Callable,
    expected: pd.Series,
) -> None:
    actual = pd.Series({file_name: fn(y=wav, **audio_params, R_compatible=True) for file_name, wav in zip(file_names, wavs)}).sort_index().astype(np.float32)
    pd.testing.assert_series_equal(expected, actual, atol=1e-2)
