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

from soundade.audio.feature.vector import Features
from soundade.data.bag import create_file_load_dictionary, load_audio_from_path, extract_features_from_audio, \
    reformat_for_dataframe, power_spectra_from_audio, log_features, transform_features, extract_banded_audio, \
    remove_dc_offset, high_pass_filter, extract_scalar_features_from_audio
# TODO This is a hack and needs to be removed
from soundade.data.filter import channel1
from soundade.data.solar import solartimes

from soundade.datasets.base import Dataset

logging.basicConfig(level=logging.INFO)

string__type = 'string'  # 'string[pyarrow]'

class ReefChorus(Dataset):
    @staticmethod
    def preprocess(b: db.Bag, save=None,
                   bands: Iterable[Tuple[int, int]] = [(50, 800), (2000, 7000), (50, 20000)]) -> db.Bag:
        b = b.map(remove_dc_offset)

        b = b.map(extract_banded_audio, bands=bands).flatten()

        if save is not None:
            logging.info(f'Saving wav files to {save}.')
            b.map(ReefChorus.write_wav, outpath=save, filename_params=['low', 'high'])

        return b

    @staticmethod
    def extract_features(b: db.Bag, frame: int, hop: int, n_fft: int, **kwargs) -> db.Bag:
        b = b.map(extract_features_from_audio, frame_length=frame, hop_length=hop, n_fft=n_fft, lim_from_dict=True,
                  **kwargs)
        return b

    @staticmethod
    def metadata(ddf: dd.DataFrame) -> dd.DataFrame:
        meta = pd.concat([ddf.dtypes.loc[:'file'], pd.Series({
            'location': 'string',
            'phase': 'string',
            'lunar cycle': 'i4',
            'timestamp': 'datetime64[ns]',
            'state': 'string',
            'hydrophone': 'string',
        }), ddf.dtypes.loc['file':].iloc[1:]])

        m = pd.DataFrame(columns=meta.index.to_list())
        m = m.astype(meta.to_dict())

        ddf = ddf.map_partitions(ReefChorus.filename_metadata, filename_column='file', meta=m)

        return ddf

    @staticmethod
    def to_dataframe(b: db.Bag, data_keys: List = None, columns_key=None) -> dd.DataFrame:
        # TODO How many columns are there?
        b = b.map(reformat_for_dataframe, data_keys=data_keys, columns_key=columns_key).flatten()

        # Sample dictionary from bag
        n = 12 * 60 * 60  # 12 samples per second * 60 seconds * 60 mintues = 43199

        meta = {
            'path': string__type,
            'file': string__type,
            'sr': 'f8',
            'low': 'f8',
            'high': 'f8',
            'frame length': 'i4',  # int
            'hop length': 'i4',  # int
            'n fft': 'i4',  # int
            'feature length': 'i4',
            'feature': string__type
        }

        meta |= {f'{i}': 'f8' for i in range(n)}

        ddf = b.to_dataframe(meta=meta)

        return ddf

    @staticmethod
    def filename_metadata(dataframe: pd.DataFrame, filename_column='file') -> pd.DataFrame:
        df = dataframe[filename_column].str.split('.', expand=True)

        df['location'] = df.loc[:, 0].str[:2]
        df['phase'] = df.loc[:, 0].str[2]
        df['lunar cycle'] = df.loc[:, 0].str[3:].astype(int)
        df['timestamp'] = pd.to_datetime(df.loc[:, 3] + df.loc[:, 1].str[:4], format='%y%m%d%H%M')
        df['state'] = df.loc[:, 1].str[-1]
        df['hydrophone'] = df.loc[:, 2]

        # Reassemble

        return dataframe.loc[:, :filename_column].join(df.drop(columns=[0, 1, 2, 3, 4])).join(
            dataframe.loc[:, filename_column:].iloc[:, 1:])
