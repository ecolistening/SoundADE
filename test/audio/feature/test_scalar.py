import pytest

import librosa
import os
import maad
import maad.sound
import numpy as np
import pandas as pd
import pathlib
import subprocess
import yaml

from soundade.audio.feature.scalar import (
    acoustic_complexity_index,
    acoustic_evenness_index,
    bioacoustic_index,
)

@pytest.fixture(scope="session", autouse=True)
def extract_seewave_features():
    subprocess.run(["Rscript", "test/helpers/seewave.R"], check=True)

@pytest.fixture
def fixtures_path():
    return pathlib.Path(os.path.dirname(__file__)).parent.parent / "fixtures"

@pytest.fixture
def fft_params(fixtures_path):
    with open(fixtures_path / "fft_params.yml", "r") as f:
        return yaml.safe_load(f.read())

@pytest.fixture
def file_paths(fixtures_path):
    return list((fixtures_path / "audio").glob("*.wav"))

window_map = {
    "hanning": "hann",
}

@pytest.fixture
def expected(fixtures_path):
    df = pd.read_parquet(fixtures_path / "indices.parquet")
    df = df.set_index("file_name").sort_index()
    return df.astype({col: np.float32 for col in df.columns})

def test_acoustic_complexity_index(file_paths, fft_params, expected):
    actual = pd.DataFrame([
        {
            "file_name": file_path.name,
            "acoustic_complexity_index": acoustic_complexity_index(
                spectrograms=maad.sound.spectrogram(
                    *librosa.load(file_path, sr=fft_params["sample_rate"]),
                    window=window_map[fft_params["window"]],
                    nperseg=fft_params["window_length"],
                    noverlap=0,
                    mode="amplitude",
                )
            )
        }
        for file_path in file_paths
    ]).set_index("file_name").sort_index()
    pd.testing.assert_frame_equal(expected, actual, atol=1e-1, rtol=1e-3)

def test_acoustic_evenness_index():
    pass

def test_bioacoustic_index():
    pass

def test_spectral_flux():
    pass

def test_zero_crossing_rate():
    pass

def test_spectral_centroid():
    pass

def test_root_mean_square():
    pass


