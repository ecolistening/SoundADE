import datetime as dt
import logging
import pandas as pd
import re

from pathlib import Path
from typing import Any, List, Dict

from soundade.datasets.base import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class NatureSense(Dataset):
    SITE_LEVEL_0: str = "nature_sense"
    SMM2_PATTERN = (
        "(?P<recorder_model>Song_Meter_Mini_2)/"
        "(?P<site_level_1>[^/]+)/"
        "[^/]+_(?P<site_level_2>[^/]+)/"
        ".*?/"
        "[^_]+_(?P<timestamp>\d{8}_\d{6})\.[wav|mp3|flac]"
    )
    AUDIO_MOTH_PATTERN = (
        "(?P<recorder_model>Audiomoth)s/"
        "(?P<site_level_1>[^/]+)/"
        "(?P<site_level_2>[^/]+_[^/]+)_\d+/"
        "(?P<timestamp>\d{8}_\d{6})\.[wav|mp4|flac]"
    )
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
