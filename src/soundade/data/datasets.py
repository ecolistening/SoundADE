import dask.bag as db
import logging
import numpy as np
import pandas as pd
import re
import soundfile as sf
from dask import dataframe as dd
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from soundade.audio.feature.vector import Features
from typing import List, Iterable, Tuple, Dict

from soundade.data import channel1
from soundade.data import create_file_load_dictionary, load_audio_from_path, extract_features_from_audio, \
    reformat_for_dataframe, power_spectra_from_audio, log_features, transform_features, extract_banded_audio, \
    remove_dc_offset, high_pass_filter, extract_scalar_features_from_audio
from soundade.data import solartimes

logging.basicConfig(level=logging.INFO)

string__type = 'string'  # 'string[pyarrow]'


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
        print('Getting file list')
        file_list = list(Path(directory).rglob('*.[wW][aA][vV]'))

        print('Creating load dictionary')
        # Break files into parts
        file_d = create_file_load_dictionary(file_list)
        print(f'Loaded {len(file_d)} files')

        file_d = list(filter(cls.prefilter_file_dictionary, file_d))
        print(f'Filtered to {len(file_d)} files')

        print('Loading files in Dask')
        # Load the files from the sequence
        b = db.from_sequence(file_d, npartitions=npartitions)
        print(f'Partitions after load: {b.npartitions}')
        b = b.map(load_audio_from_path).filter(lambda d: d is not None)
        print(f'Partitions after flatten: {b.npartitions}')

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
    def to_parquet(ddf, path: Path, **kwargs):
        dd.to_parquet(ddf, path, version='2.6', allow_truncated_timestamps=True, write_index=False, **kwargs)


class SoundingOutDiurnal(Dataset):

    @staticmethod
    def prefilter_file_dictionary(d: Dict) -> bool:
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
        # ddf = SoundingOutDiurnal.timeparts(ddf)
        ddf = SoundingOutDiurnal.country_habitat(ddf)  # , use_meta=False)

        # ddf = SoundingOutDiurnal.solar(ddf)

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
    def timeparts(df):
        df['date'] = df.timestamp.dt.date
        df['hour'] = df.timestamp.dt.hour
        df['time'] = df.timestamp.dt.time

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

    @staticmethod
    def solar(df, dask=True, use_meta=True):
        # meta = df.assign(**{
        #         'dawn': df.timestamp, 'sunrise': df.timestamp, 'noon': df.timestamp,
        #         'sunset': df.timestamp, 'dusk': df.timestamp,
        #         'hours after dawn': 0.0, 'hours after sunrise': 0.0, 'hours after noon': 0.0,
        #         'hours after sunset': 0.0, 'hours after dusk': 0.0,
        #         'dawn end': df.timestamp, 'dusk start': df.timestamp, 'dddn': ''
        #     })
        ts_type = 'M'  # ddf.dtypes.timestamp
        float_type = 'float64'
        str_type = 'string'

        metadata = df.iloc[:, :df.columns.get_loc('0')]
        features = df.iloc[:, df.columns.get_loc('0'):]

        meta = pd.concat([metadata.dtypes, pd.Series({
            'dawn': ts_type, 'sunrise': ts_type, 'noon': ts_type,
            'sunset': ts_type, 'dusk': ts_type,
            'hours after dawn': float_type, 'hours after sunrise': float_type,
            'hours after noon': float_type,
            'hours after sunset': float_type, 'hours after dusk': float_type,
            'dawn end': ts_type, 'dusk start': ts_type, 'dddn': str_type
        }), features.dtypes])

        m = pd.DataFrame(columns=meta.index.to_list())
        m = m.astype(meta.to_dict())

        if use_meta:
            df = df.map_partitions(solartimes, meta=m)
        else:
            df = df.map_partitions(solartimes)

        return df


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
            print(f'Skipping {d["path"].name}')
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
    def timeparts(df):
        df['date'] = df.timestamp.dt.date
        df['hour'] = df.timestamp.dt.hour
        df['time'] = df.timestamp.dt.time

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

    @staticmethod
    def solar(df, dask=True, use_meta=True):
        ts_type = 'M'  # ddf.dtypes.timestamp
        float_type = 'float64'
        str_type = 'string'

        meta = pd.concat([df.dtypes, pd.Series({
            'dawn': ts_type, 'sunrise': ts_type, 'noon': ts_type,
            'sunset': ts_type, 'dusk': ts_type,
            'hours after dawn': float_type, 'hours after sunrise': float_type,
            'hours after noon': float_type,
            'hours after sunset': float_type, 'hours after dusk': float_type,
            'dawn end': ts_type, 'dusk start': ts_type, 'dddn': str_type
        })])

        m = pd.DataFrame(columns=meta.index.to_list())
        m = m.astype(meta.to_dict())

        if use_meta:
            df = df.map_partitions(solartimes, meta=m)
        else:
            df = df.map_partitions(solartimes)

        return df


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


class Cairngorms(Dataset):
    @staticmethod
    def extract_features(b: db.Bag, frame: int, hop: int, n_fft: int, **kwargs) -> db.Bag:
        '''Override the default extract features to extract a single feature from each file.'''
        logging.info('Extracting Cairngorms Features')
        b = b.map(extract_scalar_features_from_audio, frame_length=frame, hop_length=hop, n_fft=n_fft, **kwargs)
        return b

    @staticmethod
    def metadata(ddf: dd.DataFrame) -> dd.DataFrame:
        meta = pd.concat([ddf.dtypes.loc[:'path'], pd.Series({
            'location': 'string',
            'timestamp': 'datetime64[ns]',
        }), ddf.dtypes.loc['path':].iloc[1:]])

        m = pd.DataFrame(columns=meta.index.to_list())
        m = m.astype(meta.to_dict())

        ddf = ddf.map_partitions(Cairngorms.filename_metadata, filename_column='path', meta=m)

        return ddf

    @staticmethod
    def filename_metadata(dataframe: pd.DataFrame, filename_column='path') -> pd.DataFrame:
        df = dataframe[filename_column].str.split('/', expand=True)

        try:
            df['location'] = df.iloc[:, -3].str.split('_', expand=True).iloc[:, -1]
            df['timestamp'] = pd.to_datetime(df.iloc[:, -2], format='%Y%m%d_%H%M%S.WAV')
        except IndexError as e:
            logging.warning(f'Error in processing metadata for {dataframe[filename_column]}')
            df['location'] = ''
            df['timestamp'] = datetime.now()

        # Reassemble
        return dataframe.loc[:, :filename_column].join(df.loc[:, 'location':'timestamp']).join(
            dataframe.loc[:, filename_column:].iloc[:, 1:])

    @staticmethod
    def to_dataframe(b: db.Bag, data_keys: List = None, columns_key=None) -> dd.DataFrame:
        '''Override the default to_dataframe to return a single row from each bag.'''
        b = b.map(reformat_for_dataframe, data_keys=data_keys, columns_key=columns_key, scalar_values=True).flatten()

        ddf = b.to_dataframe()

        return ddf


datasets = {
    'SoundingOutDiurnal': SoundingOutDiurnal,
    'SoundingOutChorus': SoundingOutChorus,
    'ReefChorus': ReefChorus,
    'Cairngorms': Cairngorms
}
