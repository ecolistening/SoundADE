import dask.bag as db
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import re
import soundfile as sf

from dask import dataframe as dd
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any, List, Iterable, Tuple, Dict

from soundade.audio.feature.vector import Features
from soundade.data.bag import (
    file_path_to_audio_dict,
    valid_audio_file,
    create_file_load_dictionary,
    load_audio_from_path,
    extract_features_from_audio,
    reformat_for_dataframe,
    power_spectra_from_audio,
    log_features,
    transform_features,
    extract_banded_audio,
    remove_dc_offset,
    high_pass_filter,
    extract_scalar_features_from_audio,
)

from soundade.data.metadata import timeparts
from soundade.data.solar import solartimes

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Dataset:
    @staticmethod
    def prefilter_file_dictionary(d: Dict) -> bool:
        return True
