import logging
import pandas as pd
import re

from dataclasses import dataclass
from pathlib import Path

from soundade.datasets.base import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@dataclass
class SoundingOutChorus(Dataset):
    PATTERN = (
        r"(?P<site_level_1>[^/]+)/"
        r"(?P<site_level_2>[^/]+)/"
        r"(?P<site_level_3>[^/]+)/"
        r"(?P<recorder_name>[A-Z]{2}-[0-9]{2})_"
        r"(?P<channel>[0-9])_"
        r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_"
        r"(?P<hour>\d{2})(?P<minute>\d{2})"
        r"\.(?i:wav|flac|mp3)"
    )
