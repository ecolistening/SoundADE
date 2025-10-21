import pathlib
import pandas as pd
import logging
import re
import soundfile

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Tuple,
)

from soundade.utils import suppress_output

__all__ = [
    "embed",
    "species_probs",
]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

_analyzer = None

BIRDNET_EMBEDDING_DIM = 1024

@suppress_output()
def _fetch_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = Analyzer()
    return _analyzer

def species_probs_meta():
    return pd.DataFrame({
        "file_id": pd.Series(dtype="string"),
        "min_conf": pd.Series(dtype="float64"),
        "model": pd.Series(dtype="object"),
        "common_name": pd.Series(dtype="object"),
        "scientific_name": pd.Series(dtype="object"),
        "label": pd.Series(dtype="object"),
        "start_time": pd.Series(dtype="float64"),
        "end_time": pd.Series(dtype="float64"),
        "confidence": pd.Series(dtype="float64"),
    })

def embed_meta():
    return pd.DataFrame({
        "file_id": pd.Series(dtype="string"),
        "model": pd.Series(dtype="object"),
        "start_time": pd.Series(dtype="float64"),
        "end_time": pd.Series(dtype="float64"),
        **{
            dim: pd.Series(dtype="float64")
            for dim in map(str, range(BIRDNET_EMBEDDING_DIM))
        },
    })

@suppress_output()
def species_probs(
    audio_dict: pd.Series,
    min_conf: float,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Returns a list of detections, each of the form:
    {
        file_id: string,
        model: string,
        common_name: string,
        scientific_name: string,
        label: string,
        start_time: float64,
        end_time: float64,
        confidence: float64,
        min_conf: float64,
    }
    """
    # lazy load analyzer on worker process (cached globally)
    analyzer = _fetch_analyzer()
    # init birdnetlib with all relevant parameters
    # TODO: add support for known species lists
    recording = Recording(
        analyzer,
        audio_dict.get("local_file_path"),
        lat=audio_dict.get("latitude"),
        lon=audio_dict.get("longitude"),
        date=ts.date() if pd.notnull(ts := audio_dict.get("timestamp")) else None,
        min_conf=min_conf,
        **kwargs,
    )
    # extract predictions
    recording.analyze()

    detections = []
    if len(recording.detections):
        for detection_dict in recording.detections:
            d = detection_dict.copy()
            d["file_id"] = audio_dict["file_id"]
            d["min_conf"] = min_conf
            d["model"] = f"BirdNET_GLOBAL_6K_V{analyzer.version}"
            detections.append(d)
    return detections

@suppress_output()
def embed(
    audio_dict: pd.Series,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Each 3s embedding is returned as a dictionary
    {
        file_id: string,
        model: string,
        start_time: float64,
        end_time: float64,
        0: float64,
        2: float64,
        ...
        1023: float64,
    }
    """
    # lazy load analyzer on worker process (cached globally)
    analyzer = _fetch_analyzer()
    # init birdnetlib with all relevant parameters
    recording = Recording(
        analyzer,
        audio_dict.get("local_file_path"),
        lat=audio_dict.get("latitude"),
        lon=audio_dict.get("longitude"),
        date=ts.date() if pd.notnull(ts := audio_dict.get("timestamp")) else None,
        **kwargs,
    )
    # extract embeddings
    recording.extract_embeddings()
    embeddings = []
    for embedding_info in recording.embeddings:
        embedding_dict = {
            "file_id": audio_dict["file_id"],
            "model": f"BirdNET_GLOBAL_6K_V{analyzer.version}",
            # concatenate timestep information, i.e. all other fields not in the embeddings info
            **{k: v for k, v in embedding_info.items() if k != "embeddings"},
            # embedding dimension column as string integers
            **{str(dim): value for dim, value in enumerate(embedding_info["embeddings"])}
        }
        embeddings.append(embedding_dict)
    return embeddings
