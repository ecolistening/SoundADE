import logging
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import List, Iterable, Tuple, Dict

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

class Kilpisjarvi(Dataset):
    @staticmethod
    def preprocess(b: db.Bag, save=None) -> db.Bag:
        b = b.map(remove_dc_offset)
        b = b.map(high_pass_filter, fcut=300, forder=2, fname='butter', ftype='highpass')

        if save is not None:
            logging.info(f'Saving wav files to {save}.')
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
        logging.info('Extracting Kilpisjarvi Features')
        b = b.map(
            extract_scalar_features_from_audio,
            frame_length=frame,
            hop_length=hop,
            n_fft=n_fft,
            **kwargs
        )
        return b

    @staticmethod
    def metadata_fields():
        return pd.Series({
            "location": "string",
            "recorder": "string",
            "timestamp": "datetime64[ns]",
            "recorder_model": "string",
        })

    @staticmethod
    def metadata(ddf: dd.DataFrame) -> dd.DataFrame:
        meta = pd.concat([
            ddf.dtypes.loc[:'path'],
            Kilpisjarvi.metadata_fields(),
            ddf.dtypes.loc["path":].iloc[1:]
        ])
        m = pd.DataFrame(columns=meta.index.to_list())
        m = m.astype(meta.to_dict())

        ddf = ddf.map_partitions(Kilpisjarvi.filename_metadata, filename_column="path", meta=m)

        return ddf

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
