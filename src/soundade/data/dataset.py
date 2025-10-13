from __future__ import annotations

import attr
import datetime as dt
import logging
import pandas as pd
import re
import yaml

from pathlib import Path
from typing import Any, Dict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEFAULT_PARAMS = dict(
    sample_rate=48_000,
    n_fft=2048,
    hop_length=1024,
    min_conf=0.0,
    segment_duration=60.0,
    bin_step=500,
    db_threshold=-47,
    R_compatible=False,
    fcut=300,
    bi_flim=[2000, 16000],
    aei_flim=[0, 20000],
)

@attr.define
class Dataset:
    name: str
    pattern: str
    sample_rate: int = attr.field(
        default=DEFAULT_PARAMS["sample_rate"],
        metadata={
            "on_default": lambda: logger.warning((
                f"No sample rate provided. Defaulting to sample_rate={DEFAULT_PARAMS['sample_rate']}."
                "Any audio below this value will be upsampled which will cause aliasing and bias results."
            ))
        },
    )
    n_fft: int = attr.field(
        default=DEFAULT_PARAMS["n_fft"],
        metadata={
            "on_default": lambda: logger.warning((
                f"No FFT window size provided. Defaulting to n_fft={DEFAULT_PARAMS['n_fft']}."
            ))
        },
    )
    hop_length: int = attr.field(
        default=DEFAULT_PARAMS["hop_length"],
        metadata={
            "on_default": lambda: logger.warning((
                f"No FFT hop size provided. Defaulting to hop_length={DEFAULT_PARAMS['min_conf']}."
            ))
        },
    )
    frame_length: int = attr.field(
        default=DEFAULT_PARAMS["n_fft"],
        metadata={
            "on_default": lambda: logger.warning((
                f"No window size provided. Defaulting to same as FFT size frame_length={DEFAULT_PARAMS['n_fft']}."
            ))
        },
    )
    min_conf: float = attr.field(
        default=DEFAULT_PARAMS["min_conf"],
        metadata={
            "on_default": lambda: logger.warning((
                f"No threshold provided for BirdNET. Defaulting to min_conf={DEFAULT_PARAMS['min_conf']}."
            ))
        },

    )
    segment_duration: float = attr.field(
        default=DEFAULT_PARAMS["segment_duration"],
        metadata={
            "on_default": lambda: logger.warning((
                f"No segment duration provided. Defaulting to segment_duration={DEFAULT_PARAMS['segment_duration']}."
            ))
        },
    )
    bin_step: float = attr.field(
        default=DEFAULT_PARAMS["bin_step"],
        metadata={
            "on_default": lambda: logger.warning((
                f"No bin step for the AEI provided. Defaulting to bin_step={DEFAULT_PARAMS['bin_step']}."
            ))
        },
    )
    db_threshold: float = attr.field(
        default=DEFAULT_PARAMS["db_threshold"],
        metadata={
            "on_default": lambda: logger.warning((
                f"No dB threshold for the AEI provided. Defaulting to db_threshold={DEFAULT_PARAMS['db_threshold']}."
            ))
        },
    )
    fcut: float = attr.field(
        default=DEFAULT_PARAMS["fcut"],
        metadata={
            "on_default": lambda: logger.warning((
                f"No high pass filter cut-off value set. Defaulting to fcut={DEFAULT_PARAMS['fcut']}"
            ))
        },
    )
    R_compatible: float = attr.field(
        default=DEFAULT_PARAMS["R_compatible"],
        metadata={
            "on_default": lambda: logger.warning((
                f"R_compatible flag set, computing seewave compatible values."
            ))
        },
    )
    bi_flim: float = attr.field(
        default=DEFAULT_PARAMS["bi_flim"],
        metadata={
            "on_default": lambda: logger.warning((
                f"BI frequeny bounds not set, defaulting to {DEFAULT_PARAMS['bi_flim']}"
            ))
        },
    )
    aei_flim: float = attr.field(
        default=DEFAULT_PARAMS["aei_flim"],
        metadata={
            "on_default": lambda: logger.warning((
                f"AEI frequeny bounds not set, defaulting to {DEFAULT_PARAMS['aei_flim']}"
            ))
        },
    )

    @classmethod
    def from_config_path(cls, config_path: Path) -> Dataset:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f.read())
        return cls(**config)

    @property
    def acoustic_feature_params(self):
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "frame_length": self.frame_length,
            "bin_step": self.bin_step,
            "db_threshold": self.db_threshold,
            "R_compatible": self.R_compatible,
            "bi_flim": self.bi_flim,
            "aei_flim": self.aei_flim,
            "compatibility": "seewave" if self.R_compatible else "QUT",
            "window": "hann",
        }

    @property
    def birdnet_params(self):
        return {
            "min_conf": self.min_conf,
        }

    def index_sites(self, root_dir: Path, sites_file: Path) -> pd.DataFrame | None:
        return pd.read_parquet(sites_file) if sites_file.exists() else None

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
