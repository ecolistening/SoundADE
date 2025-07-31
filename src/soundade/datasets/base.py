import abc
import datetime as dt
import logging
import pandas as pd
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@dataclass
class Dataset(abc.ABC):
    def index_sites(self, root_dir: str | Path) -> pd.DataFrame:
        assert (root_dir / "locations_table").exists(), \
            f"{self.__class__.__name__} locations is not extracted but provided as a separate file"
        return pd.read_parquet(root_dir / "locations_table.parquet")

    def extract_site_name(self, audio_dict: Dict[str, Any]) -> Dict[str, Any]:
        file_path = audio_dict["local_file_path"]
        match = self._get_match(file_path)
        if match is None:
            log.warning(f"Failed to extract site name on {file_path}")
            return audio_dict
        audio_dict.update({"site_name": self._extract_site_hierarchy(match)})
        return audio_dict

    def extract_timestamp(self, audio_dict: Dict[str, Any]) -> Dict[str, Any]:
        file_path = audio_dict["local_file_path"]
        match = self._get_match(file_path)
        if match is None:
            log.warning(f"Failed to extract timestamp from {file_path}")
            return audio_dict
        groups = match.groupdict()
        year, month, day = groups.get("year"), groups.get("month"), groups.get("day")
        hour, minute, second = groups.get("hour"), groups.get("minute") or "00", groups.get("second") or "00"
        timestamp = dt.datetime.strptime(f"{year}{month}{day}_{hour}{minute}{second}", "%Y%m%d_%H%M%S")
        audio_dict.update({"timestamp": timestamp})
        return audio_dict

    def _extract_site_hierarchy(self, match):
        site_levels = []
        level = 1
        while True:
            try: site_levels.append(match.group(f"site_level_{level}"))
            except: break
            level += 1
        return "/".join(site_levels)

    def _get_match(self, file_path):
        return re.search(self.PATTERN, file_path, flags=re.IGNORECASE)
