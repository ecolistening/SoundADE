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

class SoundingOutDiurnal(Dataset):
    habitat_country = {
        'BA': 'uk',
        'KN': 'uk',
        'PL': 'uk',
        'PO': 'ecuador',
        'FS': 'ecuador',
        'TE': 'ecuador',
    }

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
        try:  # Filter out short/not started at the right time files.
            # Extract the metadata
            regstr = r'(?P<location>\w+)-(?P<recorder>\d+)_(?P<channel>\d+)_(?P<timestamp>\d+_\d+)(?:_000)?.wav'
            meta_raw = re.search(regstr, d['path'].name).groupdict()
            location = meta_raw['location'][:2]
            meta = {
                'country': SoundingOutDiurnal.habitat_country[location],
                'habitat': SoundingOutDiurnal.habitat_number[location],
                'recorder': int(meta_raw['recorder'])
            }

            channel = int(meta_raw['channel'])

            if (meta in channel1 and channel == 1) or (meta not in channel1 and channel == 0):
                return True

            return False
        except ValueError as e:
                logging.info(f'Skipping {d["path"].name}')
                return False

    @staticmethod
    def preprocess(b: db.Bag, save=None) -> db.Bag:
        '''Preprocess the file names

        1. Extract filename metadata
        2. Filter unwanted channels

        :param b:
        :return:
        '''

        b = b.map(high_pass_filter, fcut=300, forder=2, fname='butter', ftype='highpass')

        if save is not None:
            logging.info(f'Saving wav files to {save}.')
            b.map(SoundingOutDiurnal.write_wav, outpath=save).compute()

        return b

    @staticmethod
    def extract_features(b: db.Bag, frame: int, hop: int, n_fft: int, **kwargs) -> db.Bag:
        logging.debug('Extracting features')
        b = b.map(extract_features_from_audio, frame_length=frame, hop_length=hop, n_fft=n_fft, **kwargs)
        logging.debug('Deciding whether to take the Log of features')
        b = b.map(log_features, features=['acoustic evenness index', 'root mean square'])
        b = b.map(transform_features, lambda f: np.log(1.0 - np.array(f)), name='log(1-{f})',
                  features=['temporal entropy'])

        return b

    @staticmethod
    def to_dataframe(b: db.Bag, data_keys: List = None, columns_key=None) -> dd.DataFrame:

        if data_keys is None:
            data_keys = [f.name for f in Features] + \
                        [f'log {f}' for f in ['acoustic evenness index', 'root mean square']] + \
                        [f'log(1-{f})' for f in ['temporal entropy']]

        b = b.map(reformat_for_dataframe, data_keys=data_keys, columns_key=columns_key).flatten()

        # Sample dictionary from bag
        n = 12 * 60  # 12 samples per second * 60 seconds

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
        ddf = SoundingOutDiurnal.filename_metadata(ddf, cols_data)
        ddf = SoundingOutDiurnal.country_habitat(ddf)
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
            lambda r: f'{r.country.upper()[:2]}{SoundingOutDiurnal.habitat_number[r.location] + 1}',
            **kwargs)

        if use_meta:
            kwargs['meta'] = (None, int)
        df['habitat'] = df.apply(lambda r: SoundingOutDiurnal.habitat_number[r.location], **kwargs)

        return df
