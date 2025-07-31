import datetime as dt
import logging
import pandas as pd
import re

from pathlib import Path
from typing import Any, List, Dict

from soundade.datasets.base import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class WildingRadio(Dataset):
    PATTERN = (
        "recordings/"
        "(?P<site_level_2>[^/]+)/"
        "(?P<year>\d{4})/"
        "(?P<month>\d{2})/"
        "(?P<day>\d{2})/"
        "(?P<site_level_1>[^-]+)-(?P<timestamp>\d{2}-\d{2}-\d{2}-\d{4})\.[wav|mp3|flac]"
    )
    TIMESTAMP_FORMAT = "%H-%d-%m-%Y"
