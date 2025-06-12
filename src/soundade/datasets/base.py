import dask.bag as db
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import re
import soundfile as sf

from dask import dataframe as dd
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any, List, Iterable, Tuple, Dict

from soundade.audio.feature.vector import Features
from soundade.data.bag import create_file_load_dictionary, load_audio_from_path, extract_features_from_audio, \
    reformat_for_dataframe, power_spectra_from_audio, log_features, transform_features, extract_banded_audio, \
    remove_dc_offset, high_pass_filter, extract_scalar_features_from_audio

# TODO This is a hack and needs to be removed
from soundade.data.filter import channel1
from soundade.data.solar import solartimes

logging.basicConfig(level=logging.INFO)

DATE32 = "date32[day]"
TIMEDELTA64 = "timedelta64[us]"
DATETIME64 = "datetime64[ns]"
string__type = 'string'  # 'string[pyarrow]'

def timeparts(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = df.timestamp.dt.date
    df['hour'] = df.timestamp.dt.hour
    df['time'] = df.timestamp.dt.time
    return df

class Dataset:
    @staticmethod
    def write_wav(d, outpath, filename_params: List = None):
        p = Path(outpath) / d['file']

        if filename_params is not None:
            p = p.with_stem(f'{p.stem}_{"_".join([d[fnp] for fnp in filename_params])}')

        logging.info(f'Saving file {d["path"]} to {p}')

        sf.write(p, d['audio'], d['sr'], subtype='FLOAT')

    @staticmethod
    def prefilter_file_dictionary(d: Dict) -> bool:
        return True

    @classmethod
    def load(cls, directory, partition_size=None, npartitions=None) -> db.Bag:
        '''Load audio files from a directory into a Dask Bag.

        If any file selection can be performed before audio loading, it should be done here.

        Bag entries are dictionaries with the format:
        data_dict = {
            'path': file path [string],
            'file': file name [string],
            'audio': raw audio [np.array],
            'sr': sample rate [int]
        }

        :param directory:
        :param npartitions:
        :return:
        '''
        logging.info('Getting file list')
        file_list = list(Path(directory).rglob('*.[wW][aA][vV]'))

        logging.info('Creating load dictionary')
        # Break files into parts
        file_d = create_file_load_dictionary(file_list)
        logging.info(f'Loaded {len(file_d)} files')

        file_d = list(filter(cls.prefilter_file_dictionary, file_d))
        logging.info(f'Filtered to {len(file_d)} files')

        logging.info('Loading files in Dask')
        # Load the files from the sequence
        b = db.from_sequence(file_d, npartitions=npartitions)
        logging.info(f'Partitions after load: {b.npartitions}')
        b = b.map(load_audio_from_path).filter(lambda d: d is not None)
        logging.info(f'Partitions after flatten: {b.npartitions}')

        return b

    @staticmethod
    def preprocess(b: db.Bag, save=None) -> db.Bag:
        '''Performs any necessary preprocessing on audio data.

        This can include:
        * audio preprocessing :
            * high/low/bandpass/etc filtering
        * file preprocessing:
            * filtering
            * selection

        :return:
        '''
        return b

    @staticmethod
    def power_spectra(b: db.Bag, **kwargs):
        b = b.map(power_spectra_from_audio, **kwargs)
        return b

    @staticmethod
    def extract_features(b: db.Bag, frame: int, hop: int, n_fft: int, **kwargs) -> db.Bag:
        '''Extract features from the raw audio and discard audio.

        Returns a bag with entries in the format
        data_dict = {
            'path': file path [string],
            'file': file name [string],
            'sr': sample rate [int]
            'frame length': frame length in samples [int],
            'hop length': hop length in samples [int],
            'n fft': number of samples per fft [int],
            'FEATURE_NAME': 'FEATURE_ARRAY' [for each feature computed]
        })

        :param b:
        :param frame:
        :param hop:
        :param n_fft:
        :param kwargs:
        :return:
        '''
        b = b.map(extract_features_from_audio, frame_length=frame, hop_length=hop, n_fft=n_fft, **kwargs)
        return b

    @staticmethod
    def to_dataframe(b: db.Bag, data_keys: List = None, columns_key=None) -> dd.DataFrame:
        b = b.map(reformat_for_dataframe, data_keys=data_keys, columns_key=columns_key).flatten()

        ddf = b.to_dataframe()

        return ddf

    @staticmethod
    def metadata(ddf: dd.DataFrame) -> dd.DataFrame:
        return ddf

    @staticmethod
    def to_parquet(ddf: dd.DataFrame, path: Path, compute=False, **kwargs):
        if compute:
            ddf.compute().to_parquet(path, **kwargs)
        else:
            dd.to_parquet(ddf, path, version='2.6', allow_truncated_timestamps=True, write_index=False, **kwargs)

    @staticmethod
    def timeparts(
        ddf: dd.DataFrame
    ) -> dd.DataFrame:
        meta = pd.concat([ddf.dtypes, pd.Series({
            "date": DATETIME64,
            "hour": np.int8,
            "time": TIMEDELTA64,
        })])
        ddf = ddf.map_partitions(
            timeparts,
            meta=pd.DataFrame(columns=meta.index.to_list()).astype(meta.to_dict()),
        )
        return ddf

    @staticmethod
    def solar(
        ddf: dd.DataFrame,
        sitesfile: str
    ) -> dd.DataFrame:
        meta = pd.concat([ddf.dtypes, pd.Series({
            "location_id": str,
            "latitude": np.float64,
            "longitude": np.float64,
            "country": str,
            "timezone": str,
            'dawn': DATETIME64,
            'sunrise': DATETIME64,
            'noon': DATETIME64,
            'sunset': DATETIME64,
            'dusk': DATETIME64,
            'hours after dawn': np.float64,
            'hours after sunrise': np.float64,
            'hours after noon': np.float64,
            'hours after sunset': np.float64,
            'hours after dusk': np.float64,
            'dawn end': DATETIME64,
            'dusk start': DATETIME64,
            'dddn': str,
        })])
        ddf = ddf.map_partitions(
            solartimes,
            locations=sitesfile,
            meta=pd.DataFrame(columns=meta.index.to_list()).astype(meta.to_dict()),
        )
        return ddf

