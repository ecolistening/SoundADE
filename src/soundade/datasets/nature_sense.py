import logging
import datetime as dt
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import (
    Any,
    List,
    Iterable,
    Tuple,
    Dict,
)

import dask.bag as db
import numpy as np
import pandas as pd
import soundfile as sf
from dask import dataframe as dd

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

class NatureSense(Dataset):
    @staticmethod
    def index_sites(root_dir: str | Path) -> pd.DataFrame:
        assert (root_dir / "locations_table").exists(), "NatureSense locations is not extracted but provided as a separate file"
        return pd.read_parquet(root_dir / "locations_table.parquet")

    @staticmethod
    def preprocess(b: db.Bag, save=None) -> db.Bag:
        b = b.map(remove_dc_offset)
        b = b.map(high_pass_filter, fcut=300, forder=2, fname='butter', ftype='highpass')

        if save is not None:
            log.info(f'Saving wav files to {save}.')
            b.map(Dataset.write_wav, outpath=save).compute()

        return b

    @staticmethod
    def to_dataframe(b: db.Bag, data_keys: List = None, columns_key=None) -> dd.DataFrame:
        '''Override the default to_dataframe to return a single row from each bag.'''
        b = b.map(
            reformat_for_dataframe,
            data_keys=data_keys,
            columns_key=columns_key,
            scalar_values=True
        ).flatten()
        ddf = b.to_dataframe()
        return ddf

    @staticmethod
    def extract_features(b: db.Bag, frame: int, hop: int, n_fft: int, **kwargs) -> db.Bag:
        '''Override the default extract features to extract a single feature from each file.'''
        log.info('Extracting NatureSense Features')
        b = b.map(
            extract_scalar_features_from_audio,
            frame_length=frame,
            hop_length=hop,
            n_fft=n_fft,
            **kwargs
        )
        return b

    @staticmethod
    def metadata(ddf: dd.DataFrame) -> dd.DataFrame:
        meta = pd.concat([
            ddf.dtypes.loc[:'path'],
            pd.Series({
                "recorder_model": "string",
                "location": "string",
                "site": "string",
                "timestamp": "datetime64[ns]",
                "site_name": "string",
            }),
            ddf.dtypes.loc["path":].iloc[1:]
        ])

        ddf = ddf.map_partitions(
            NatureSense.filename_metadata,
            filename_column="path",
            meta=meta, # TODO: check works
        )

        return ddf

    @staticmethod
    def extract_site_name(audio_dict: Dict[str, Any]) -> Dict[str, Any]:
        file_path = audio_dict["local_file_path"]
        audio_moth_match = re.search(r"Audiomoths/Audiomoths/([^/]+)/(.+?)_\D{8}", file_path)
        song_meter_match = re.search(r"Song_Meter_Mini/([^/]+)/[^/]+_([^/_]+)", file_path)
        match = None
        if audio_moth_match:
            recorder_model = "AudioMoth"
            match = audio_moth_match
        elif song_meter_match:
            recorder_model = "SMM2"
            match = song_meter_match
        if match is None:
            log.warning(f"Failed to extract site name on {file_path}")
            return audio_dict
        site_name = "/".join([match.group(1), match.group(2)])
        audio_dict.update({"site_name": site_name, "recorder_model": recorder_model })
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
            r".*?/" # match any prefix path
            "(?P<recorder_model>[^/]+)/" # extract recorder model, e.g. Audiomoths, Song_Meter_Mini
            "(?P<location>[^/]+)/" # location, e.g. Knepp, Gravetye, etc
            r"(?P<site>[^/_]+_[^/_]+)_(?:[^/]+)/" # site, e.g. N_SE1, S_SW3
            "(?P<timestamp>\d{8}_\d{6})\.wav", # timestamp for the format 20250522_123456
            expand=True,
            flags=re.IGNORECASE,
        )
        # strip trailing 's' and sub out underscores for better presentation
        metadata["recorder_model"] = metadata["recorder_model"].str.rstrip("s").replace("_", " ")
        # site_name is used as a site hierarchy in the dashboard
        metadata["site_name"] = metadata["location"] + "/" + metadata["site"]
        # timestamp always necessary, only consistent to Audiomoth format the moment
        metadata["timestamp"] = pd.to_datetime(metadata["timestamp"], format="%Y%m%d_%H%M%S")
        # TODO: why do this? surely the index is preserved? to preserve column order?
        # Just return the relevant information and join the columns after mapping
        return (
            df.loc[:, :filename_column]
            .join(metadata.loc[:, "recorder_model":"timestamp"])
            .join(df.loc[:, filename_column:].iloc[:, 1:])
        )

    @staticmethod
    def to_dataframe(b: db.Bag, data_keys: List = None, columns_key=None) -> dd.DataFrame:
        '''Override the default to_dataframe to return a single row from each bag.'''
        b = b.map(
            reformat_for_dataframe,
            data_keys=data_keys,
            columns_key=columns_key,
            scalar_values=True
        ).flatten()

        ddf = b.to_dataframe()

        return ddf
