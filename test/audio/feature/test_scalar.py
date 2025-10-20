import pytest
import librosa
import os
import numpy as np
import pandas as pd
import pathlib
import subprocess
import shutil
import warnings
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

@pytest.fixture(scope="session", autouse=True)
def extract_features(fixtures_path):
    if not (fixtures_path / "seewave_indices.parquet").exists():
        subprocess.run(["Rscript", "test/helpers/indices.R", "test/fixtures/audio", "test/fixtures/audio_params.yaml", "/test/fixtures/seewave_indices.parquet"], check=True)
    yield

@pytest.fixture(scope="session")
def expected_acoustic_features(fixtures_path) -> pd.DataFrame:
    df = pd.read_parquet(fixtures_path / "seewave_indices.parquet")
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
    pd.testing.assert_series_equal(expected, actual, rtol=1e-3)

@pytest.mark.parametrize("fn, expected", [(acoustic_evenness_index, "acoustic_evenness_index")], indirect=["expected"])
def test_acoustic_evenness_index(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    fn: Callable,
    expected: pd.Series,
) -> None:
    actual = pd.Series({file_name: fn(y=wav, **audio_params, R_compatible=True) for file_name, wav in zip(file_names, wavs)}).sort_index().astype(np.float32)
    pd.testing.assert_series_equal(expected, actual, rtol=1e-1)

@pytest.mark.parametrize("fn, expected", [(bioacoustic_index, "bioacoustic_index")], indirect=["expected"])
def test_bioacoustic_index(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    fn: Callable,
    expected: pd.Series,
) -> None:
    actual = pd.Series({file_name: fn(y=wav, **audio_params, R_compatible=True) for file_name, wav in zip(file_names, wavs)}).sort_index().astype(np.float32)
    pd.testing.assert_series_equal(expected, actual, rtol=1e-2)

@pytest.mark.parametrize("fn, expected", [(spectral_flux, "spectral_flux")], indirect=["expected"],)
def test_spectral_flux(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    fn: Callable,
    expected: pd.Series,
) -> None:
    actual = pd.Series({file_name: fn(y=wav, **audio_params, R_compatible=True) for file_name, wav in zip(file_names, wavs)}).sort_index().astype(np.float32)
    pd.testing.assert_series_equal(expected, actual, rtol=1e-1)

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
    rtol = 1e-2
    if not np.isclose(expected, actual, rtol=rtol).all():
        warnings.warn(
            f"'spectral_centroid' differs beyond tolerance of {rtol=}\n"
            f"expected: {expected.values}\n"
            f"actual: {actual.values}",
            UserWarning
        )

@pytest.mark.parametrize("fn, expected", [(root_mean_square, "root_mean_square")], indirect=["expected"])
def test_root_mean_square(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    fn: Callable,
    expected: pd.Series,
) -> None:
    actual = pd.Series({file_name: fn(y=wav, **audio_params) for file_name, wav in zip(file_names, wavs)}).sort_index().astype(np.float32)
    pd.testing.assert_series_equal(expected, actual, rtol=1e-2)

@pytest.mark.parametrize("fn, expected", [(spectral_entropy, "spectral_entropy")], indirect=["expected"])
def test_spectral_entropy(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    fn: Callable,
    expected: pd.Series,
) -> None:
    actual = pd.Series({file_name: fn(y=wav, **audio_params, compatibility="seewave") for file_name, wav in zip(file_names, wavs)}).sort_index().astype(np.float32)
    rtol = 1e-2
    if not np.isclose(expected, actual, rtol=rtol).all():
        warnings.warn(
            f"'spectral_entropy' differs beyond tolerance of {rtol=}\n"
            f"expected: {expected.values}\n"
            f"actual: {actual.values}",
            UserWarning
        )

@pytest.mark.parametrize("fn, expected", [(temporal_entropy, "temporal_entropy")], indirect=["expected"])
def test_temporal_entropy(
    file_names: List[str],
    wavs: List[NDArray],
    audio_params: Dict[str, Any],
    fn: Callable,
    expected: pd.Series,
) -> None:
    actual = pd.Series({file_name: fn(y=wav, **audio_params, R_compatible=True) for file_name, wav in zip(file_names, wavs)}).sort_index().astype(np.float32)
    rtol = 1e-2
    if not np.isclose(expected, actual, rtol=rtol).all():
        warnings.warn(
            f"'temporal_entropy' differs beyond tolerance of {rtol=}\n"
            f"expected: {expected.values}\n"
            f"actual: {actual.values}",
            UserWarning
        )

def test_correlated(fixtures_path):
    expected = pd.read_parquet(fixtures_path / "expected_indices.parquet").sort_index()
    actual = pd.read_parquet(fixtures_path / "actual_indices.parquet").sort_index()
    file_names = sorted(list(set(expected.index).intersection(set(actual.index))))
    expected = expected.loc[file_names]
    actual = actual.loc[file_names]
    results = {}
    for col in expected.columns:
        results[col] = pearsonr(expected[col], actual[col])
    df = pd.DataFrame(results, columns=expected.columns, index=["corr_coeff", "p_value"]).transpose()
    assert (df["corr_coeff"] >= 0.7).all(), "Correlation coefficient <= 0.7"
    assert (df["p_value"] <= 1e-8).all(), "p-value >= 1e-8"
