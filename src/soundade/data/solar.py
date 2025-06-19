import logging
from pathlib import Path

import pandas as pd
from astral import LocationInfo
from astral.sun import sun

from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO)

def find_sun(
    r: pd.Series,
    locations: pd.DataFrame,
) -> Dict[str, Any]:
    loc = LocationInfo(
        name=r.location,
        timezone=r.timezone,
        latitude=r.latitude,
        longitude=r.longitude
    )
    s = sun(loc.observer, r.date, tzinfo=loc.tzinfo)
    return dict([(k, s[k].replace(tzinfo=None)) for k in s])

tod_cols = ['dawn', 'sunrise', 'noon', 'sunset', 'dusk']

def solartimes(
    dataframe: pd.DataFrame,
    locations: pd.DataFrame | Path | str,
    join_columns: List[str] = ["location", "site"],
) -> pd.DataFrame:
    """
    Calculate solar event times (dawn, sunrise, sunset, dusk) for each timestamp in the given dataframe.

    Args:
        dataframe (pd.DataFrame): The input dataframe containing timestamps and other data.
        locations (pd.DataFrame | Path | str): The locations dataframe or path to the locations file.

    Returns:
        pd.DataFrame: The dataframe with solar times and other calculated features.

    Raises:
        FileNotFoundError: If the locations file is not found.
        AssertionError: If locations is not a pandas dataframe

    Notes:
        - The dataframe should have a column named 'timestamp' containing the timestamps.
        - The dataframe may contain duplicate indices, which will be reset.
        - The dataframe should have metadata columns before the 'timestamp' column.
        - The locations dataframe should have columns 'latitude', 'longitude', and 'timezone'.

    Example:
        >>> dataframe = pd.DataFrame({'timestamp': ['2022-01-01 12:00:00', '2022-01-01 13:00:00'], 'location': ['A', 'B']})
        >>> locations = pd.DataFrame({'location': ['A', 'B'], 'latitude': [40.7128, 34.0522], 'longitude': [-74.0060, -118.2437], 'timezone': ['America/New_York', 'America/Los_Angeles']})
        >>> solartimes(dataframe, locations)
    """
    # There may be duplicate indices in the dataframe, so we reset the indices,
    # creating a column called 'index' that can be used to join metadata and features later
    dataframe = dataframe.reset_index()

    # load locations if necessary
    if isinstance(locations, Path) or isinstance(locations, str):
        locations = pd.read_parquet(locations)

    assert type(locations) == pd.DataFrame, "'locations' is not a pandas DataFrame"

    # TODO: add validation for the relevant required columns in locations
    # merge location information
    df = dataframe.merge(locations, on=join_columns, how="left")

    # Replace the location/recorder/date metadata
    df = df.join(df.apply(find_sun, locations=locations, axis=1, result_type='expand'))

    # Convert to hours past event format
    relative_to_time_columns = [f'hours after {t}' for t in tod_cols]
    times = df.timestamp.to_numpy().reshape(-1, 1)
    df[relative_to_time_columns] = df[tod_cols].subtract(times, axis=0).div(pd.Timedelta(hours=1)).mul(-1)

    # Grouping into dawn/day/dusk/night
    df['dawn end'] = df.sunrise.add(df.sunrise.subtract(df.dawn))
    df['dusk start'] = df.sunset.subtract(df.dusk.subtract(df.sunset))

    df['dddn'] = 'day'
    df['dddn'] = df['dddn'].mask(~df.timestamp.between(df.dawn, df.dusk), 'night')
    df['dddn'] = df['dddn'].mask(df.timestamp.between(df.dawn, df.sunrise + (df.sunrise - df.dawn)), 'dawn')
    df['dddn'] = df['dddn'].mask(df.timestamp.between(df.sunset - (df.dusk - df.sunset), df.dusk), 'dusk')

    # drop merge columns used to calculate solar information, preserve join columns for downstream join
    df = df.drop([col for col in locations.columns if col not in join_columns], axis=1)

    # The original index is reset as map_partitions assumes that the index does not change.
    return df.set_index('index')
