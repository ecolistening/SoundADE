import datetime as dt
import logging
import pandas as pd
import re

from pathlib import Path
from typing import Any, List, Dict

from soundade.datasets.base import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Cairngorms(Dataset):
    SITE_LEVEL_0: str = "cairngorms"
    PATTERN: str = (
        "(?<recorder_model>[^/]+)_(?P<site_level_1>[^/]+)/"
        "(?P<date>\d{8})/"
        "(?P<timestamp>\d{8}_\d{6})\.[wav|flac|mp3]"
    )
    TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"
