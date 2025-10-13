import pytest

import os
import pathlib
import yaml

from typing import Any, Dict, List

@pytest.fixture(scope="session")
def fixtures_path() -> pathlib.Path:
    return pathlib.Path(os.path.dirname(__file__)) / "fixtures"

@pytest.fixture(scope="session")
def audio_params(fixtures_path) -> Dict[str, Any]:
    with open(fixtures_path / "audio_params.yml", "r") as f:
        params = {}
        for k, v in yaml.safe_load(f.read()).items():
            if type(v) == list:
                v = tuple(v)
            params[k] = v
        return params

@pytest.fixture(scope="session")
def file_paths(fixtures_path) -> List[pathlib.Path]:
    file_paths = list((fixtures_path / "audio").glob("*.wav"))
    assert len(file_paths), "No files are in the fixtures test/fixtures/audio"
    return file_paths

@pytest.fixture(scope="session")
def file_names(file_paths) -> List[str]:
    return [p.name for p in file_paths]

