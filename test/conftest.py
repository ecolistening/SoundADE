import pytest

import librosa
import os
import pathlib
import yaml

from numpy.typing import NDArray
from typing import Any, Dict, List

@pytest.fixture(scope="session")
def fixtures_path() -> pathlib.Path:
    return pathlib.Path(os.path.dirname(__file__)) / "fixtures"

@pytest.fixture(scope="session")
def config_path(fixtures_path):
    return fixtures_path / "audio_params.yml"

@pytest.fixture(scope="session")
def audio_params(config_path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        params = {}
        for k, v in yaml.safe_load(f.read()).items():
            if type(v) == list:
                v = tuple(v)
            params[k] = v
        return params

@pytest.fixture(scope="session")
def file_paths(fixtures_path) -> List[pathlib.Path]:
    file_paths = list((fixtures_path / "audio" / "SO").rglob("*.[wW][aA][vV]"))
    assert len(file_paths), "No files are in the fixtures test/fixtures/audio"
    return file_paths

@pytest.fixture(scope="session")
def file_names(file_paths) -> List[str]:
    return [p.name for p in file_paths]

@pytest.fixture(scope="session")
def wavs(file_paths, audio_params) -> List[NDArray]:
    return [librosa.load(file_path, sr=audio_params["sr"])[0] for file_path in file_paths]
