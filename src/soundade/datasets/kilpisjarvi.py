import logging
import pandas as pd
import re

from dataclasses import dataclass
from pathlib import Path

from soundade.datasets.base import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@dataclass
class Kilpisjarvi(Dataset):
    PATTERN = (
        "(?P<site_level_1>[^/_]+)(?:_\d+)?/"
        "(?:Data/)?"
        "(?P<recorder>SMA\d{5})_"
        "(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_"
        "(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})"
        "\.[wav|flac|mp3]+"
    )
