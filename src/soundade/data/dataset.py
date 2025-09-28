from __future__ import annotations

import datetime as dt
import logging
import pandas as pd
import re
import yaml

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@dataclass
class Dataset:
    name: str
    pattern: str
    sample_rate: int
    frame_length: int
    hop_length: int
    n_fft: int
    min_conf: float
    segment_duration: float

    @classmethod
    def from_config_path(cls, config_path: Path) -> Dataset:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f.read())
        return cls(**config)

    @property
    def audio_params(self):
        return {
            "sr": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "frame_length": self.frame_length,
        }

    @property
    def birdnet_params(self):
        return {
            "min_conf": self.min_conf,
        }

    def index_sites(self, root_dir: Path, sites_file: Path) -> pd.DataFrame:
        assert sites_file.exists(), \
            f"{self.__class__.__name__} locations is not extracted but provided as a separate file at {sites_file}"
        return pd.read_parquet(sites_file)

    def extract_site_name(self, audio_dict: Dict[str, Any]) -> Dict[str, Any]:
        file_path = audio_dict["file_path"]
        match = self._get_match(file_path)
        if match is None:
            log.warning(f"Failed to extract site name on {file_path}")
            return audio_dict
        site_name, site_levels = self._extract_site_hierarchy(match)
        audio_dict.update({
            "site_name": site_name,
            **{f"sitelevel_{i + 1}": level for i, level in enumerate(site_levels)}
        })
        return audio_dict

    def extract_timestamp(self, audio_dict: Dict[str, Any]) -> Dict[str, Any]:
        file_path = audio_dict["file_path"]
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
        return "/".join(site_levels), site_levels

    def _get_match(self, file_path):
        return re.search(self.pattern, file_path, flags=re.IGNORECASE)
