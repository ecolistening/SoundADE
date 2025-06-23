import dask.bag as db
import datetime as dt
import logging
import numpy as np
import pandas as pd
import re
import soundfile as sf
import uuid

from dask import dataframe as dd
from importlib.resources import files
from pathlib import Path
from typing import (
    Any,
    List,
    Iterable,
    Tuple,
    Dict,
)

from soundade.data.bag import (
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
from soundade.datasets.base import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Kilpisjarvi(Dataset):
    @staticmethod
    def index_sites(root_dir: str | Path) -> pd.DataFrame:
        """
        Index sites by extracting co-ordinate data from SongMeterMini's Summary.txt file
        """
        recorder_id_regex = re.compile("^([A-Z]{3}\d{5})_Summary.txt$")
        hemisphere_to_sign = {"N": 1, "S": -1, "E": 1, "W": -1}
        site_data = []
        for file_path in root_dir.rglob("*"):
            match = recorder_id_regex.match(file_path.name)
            if not match:
                continue
            try:
                row = next(pd.read_csv(file_path, chunksize=1)).iloc[0]
                latitude, lat_hemi, longitude, lon_hemi = row[2:6]
                site_data.append({
                    "site_id": str(uuid.uuid4()),
                    "site_name": f"{file_path.parent.name}/{match.group(1)}",
                    "location": file_path.parent.name,
                    "recorder": match.group(1),
                    "latitude": float(latitude) * hemisphere_to_sign[str(lat_hemi).strip()],
                    "longitude": float(longitude) * hemisphere_to_sign[str(lon_hemi).strip()],
                    "country": "Finland",
                    "timezone": "Europe/Helsinki",
                })
            except Exception as e:
                log.warning(e)
                continue
        return pd.DataFrame(site_data)

    @staticmethod
    def extract_site_name(audio_dict: Dict[str, Any]) -> Dict[str, Any]:
        file_path = audio_dict["local_file_path"]
        match = re.search(r"([^/]+)/Data/(SMA\d+)", file_path, re.IGNORECASE)
        if match is None:
            log.warning(f"Failed to extract site name on {file_path}")
            return audio_dict
        site_name = f"{match.group(1)}/{match.group(2)}"
        audio_dict.update({
            "site_name": site_name,
        })
        return audio_dict

    def extract_timestamp(audio_dict: Dict[str, Any]) -> Dict[str, Any]:
        file_path = audio_dict["local_file_path"]
        match = re.search(r"(\d{8}_\d{6})\.wav", file_path, re.IGNORECASE)
        if match is None:
            log.warning(f"Failed to extract timestamp from {file_path}")
            return audio_dict
        timestamp = dt.datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        audio_dict.update({
            "timestamp": timestamp,
        })
        return audio_dict

    @staticmethod
    def filename_metadata(df: pd.DataFrame, filename_column="path") -> pd.DataFrame:
        metadata = df[filename_column].str.extract(
            r"(?P<location>K\d+)/Data/(?P<recorder>\w+)_(?P<timestamp>\d+_\d+)\.wav",
            expand=True,
            flags=re.IGNORECASE,
        )
        metadata["timestamp"] = pd.to_datetime(metadata["timestamp"], format="%Y%m%d_%H%M%S")
        metadata["recorder_model"] = metadata["recorder"].str[:3]
        return (
            df.loc[:, :filename_column]
            .join(metadata.loc[:, "location":"recorder_model"])
            .join(df.loc[:, filename_column:].iloc[:, 1:])
        )
