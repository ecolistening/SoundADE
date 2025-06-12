import pandas as pd
import pyarrow as pa
import numpy as np

def filename(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''Extracts location, recorder, channel, and timestamp from filename and adds them as dataframe metadata.

    :param dataframe: the dataframe to process
    :return: dataframe with new metadata
    '''
    filename_metadata = dataframe['file'].str.extract(
        r'(?P<location>\w+)-(?P<recorder>\d+)_(?P<channel>\d+)_(?P<timestamp>\d+_\d+)(?:_000)?.wav')

    df = pd.concat([dataframe.iloc[:, :3], filename_metadata[['location', 'recorder', 'channel']],
                    pd.to_datetime(filename_metadata['timestamp'].str[:13], format='%Y%m%d_%H%M'), dataframe.iloc[:, 3:]],
                   axis=1)  # , ignore_index=True)
    df.location = df.location.str[:2]

    return df.astype({'recorder': "Int64", 'channel': "Int64"})

def temporal_categories(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''Adds dawn/dusk data for a dataframe of dawn and dusk choruses.

    :param dataframe:
    :return:
    '''

    dataframe['dawndusk'] = dataframe.timestamp.apply(lambda t: 'dawn' if t.hour < 12 else 'dusk')
    dataframe['date'] = dataframe.timestamp.dt.date
    dataframe['hour'] = dataframe.timestamp.dt.hour

    return dataframe

def timeparts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts date, time and hour from timestamp as individual columns
    """
    return pd.DataFrame({
        "date": pd.to_datetime(df.timestamp.dt.date),
        "hour": df.timestamp.dt.hour.astype(np.int8),
    }, index=df.index)

