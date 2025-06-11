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

class SoundingOutChorus(Dataset):
    habitat_number = {
        'BA': 2,
        'KN': 1,
        'PL': 0,
        'PO': 2,
        'FS': 1,
        'TE': 0,
    }

    @staticmethod
    def prefilter_file_dictionary(d: Dict) -> bool:
        gdm_f = files('ecolistening.data').joinpath('good_dates_and_mics.parquet')
        gdm = pd.read_parquet(gdm_f)
        gdm['date'] = gdm.date.dt.date

        try:  # Filter out short/not started at the right time files.
            # Extract the metadata
            regstr = r'(?P<location>\w+)-(?P<recorder>\d+)_(?P<channel>\d+)_(?P<timestamp>\d+_\d+)(?:_000)?.wav'
            meta_raw = re.search(regstr, d['path'].name).groupdict()

            location = meta_raw['location'][:2]
            meta = {
                'location': [location],
                'recorder': [int(meta_raw['recorder'])],
                'channel': [int(meta_raw['channel'])],
                'date': [datetime.strptime(meta_raw['timestamp'], '%Y%m%d_%H%M').date()]
            }

        except ValueError as e:
            logging.info(f'Skipping {d["path"].name}')
            return False

        merge = pd.merge(gdm, pd.DataFrame.from_dict(meta), on=['location', 'recorder', 'channel', 'date'], how='left',
                         indicator='exists')
        return (merge['exists'] == 'both').sum() > 0

    @staticmethod
    def extract_features(b: db.Bag, frame: int, hop: int, n_fft: int, **kwargs) -> db.Bag:
        logging.debug('Extracting features')
        b = b.map(extract_features_from_audio, frame_length=frame, hop_length=hop, n_fft=n_fft, **kwargs)
        logging.debug('Deciding whether to take the Log of features')
        b = b.map(log_features, features=['acoustic evenness index', 'root mean square'])
        b = b.map(transform_features, lambda f: np.log(1.0 - np.array(f)), name='log(1-f)',
                  features=['temporal entropy'])

        return b

    @staticmethod
    def to_dataframe(b: db.Bag, data_keys: List = None, columns_key=None) -> dd.DataFrame:
        b = b.map(reformat_for_dataframe, data_keys=data_keys, columns_key=columns_key).flatten()

        # Sample dictionary from bag
        n = 12 * 9060  # 12 samples per second * a little over 2.5 hours in seconds

        meta = {
            'path': string__type,
            'file': string__type,
            'sr': 'f8',
            'frame length': 'i4',  # int
            'hop length': 'i4',  # int
            'n fft': 'i4',  # int
            'feature': string__type
        }

        meta |= {f'{i}': 'f8' for i in range(n)}

        ddf = b.to_dataframe(meta=meta)

        return ddf

    @staticmethod
    def metadata(ddf: dd.DataFrame) -> dd.DataFrame:
        # Assume dataframe ends with data columns
        cols_data = list(ddf.loc[:, '0':].columns)

        ddf = SoundingOutChorus.filename_metadata(ddf, cols_data)
        # ddf = SoundingOutChorus.timeparts(ddf)
        ddf = SoundingOutChorus.country_habitat(ddf)  # , use_meta=False)

        # TODO: This might still be broken. Test!
        # ddf = SoundingOutChorus.solar(ddf)

        # Move meta columns to beginning
        meta_cols = ddf.drop(columns=cols_data).columns
        ddf = ddf[list(meta_cols) + cols_data]

        return ddf

    @staticmethod
    def filename_metadata(df, cols_data):
        # Extract metadata
        filename_metadata = df['file'].str.extract(
            r'(?P<location>\w+)-(?P<recorder>\d+)_(?P<channel>\d+)_(?P<timestamp>\d+_\d+)(?:_000)?.wav')

        # Organise metadata columns
        df[['location', 'recorder', 'channel']] = filename_metadata[['location', 'recorder', 'channel']]
        df['timestamp'] = dd.to_datetime(filename_metadata['timestamp'].str[:13], format='%Y%m%d_%H%M')
        df = df[['path', 'file', 'location', 'recorder', 'channel', 'timestamp', 'feature'] + cols_data]

        # Fix columns
        df.location = df.location.str[:2]
        df.recorder = df.recorder.astype('Int32')
        df.channel = df.channel.astype('Int32')

        return df

    @staticmethod
    def country_habitat(df, use_meta=True):
        kwargs = {'axis': 1}
        if use_meta:
            kwargs['meta'] = (None, str)

        df['country'] = df.apply(lambda r: 'uk' if 'UK' in r.path else 'ecuador', **kwargs)
        df['habitat code'] = df.apply(
            lambda r: f'{r.country.upper()[:2]}{SoundingOutChorus.habitat_number[r.location] + 1}',
            **kwargs)

        if use_meta:
            kwargs['meta'] = (None, int)
        df['habitat'] = df.apply(lambda r: SoundingOutChorus.habitat_number[r.location], **kwargs)

        return df

