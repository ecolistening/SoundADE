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

class WildlifeAcoustics(Dataset):
    @staticmethod
    def extract_features(b: db.Bag, frame: int, hop: int, n_fft: int, **kwargs) -> db.Bag:
        '''Override the default extract features to extract a single feature from each file.'''
        logging.info('Extracting WildlifeAcoustics Descriptors')
        b = b.map(extract_scalar_features_from_audio, frame_length=frame, hop_length=hop, n_fft=n_fft, **kwargs)
        return b

    @staticmethod
    def metadata(ddf: dd.DataFrame) -> dd.DataFrame:
        meta = pd.concat([ddf.dtypes.loc[:'path'], pd.Series({
            'recorder model': 'string',
            'recorder serial': 'string',
            'timestamp': 'datetime64[ns]',
        }), ddf.dtypes.loc['path':].iloc[1:]])

        m = pd.DataFrame(columns=meta.index.to_list())
        m = m.astype(meta.to_dict())

        ddf = ddf.map_partitions(WildlifeAcoustics.filename_metadata, filename_column='path', meta=m)

        return ddf

    @staticmethod
    def filename_metadata(dataframe: pd.DataFrame, filename_column='path') -> pd.DataFrame:
        df = dataframe[filename_column].str.split('/', expand=True)
        filename = df.iloc[:, -1]
        rec = filename.str.split('_', expand=True)

        # try:
        # Split the file name by '_', the first
        df['recorder model'] = rec.iloc[:, 0].str[:3]
        df['recorder serial'] = rec.iloc[:, 0].str[3:]
        df['timestamp'] = pd.to_datetime(filename.str.extract(r'(\d{8}_\d{6})').loc[:, 0], format='%Y%m%d_%H%M%S')
        # except IndexError as e:
        #     logging.warning(f'Error in processing metadata for {dataframe[filename_column]}')
        #     df['location'] = ''
        #     df['timestamp'] = datetime.now()

        # Reassemble
        return dataframe.loc[:, :filename_column].join(df.loc[:, 'recorder model':'timestamp']).join(
            dataframe.loc[:, filename_column:].iloc[:, 1:])
