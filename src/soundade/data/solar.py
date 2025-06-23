import logging
import pandas as pd
import numpy as np
import uuid

from astral import LocationInfo
from astral.sun import sun
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

tod_cols = ['dawn', 'sunrise', 'noon', 'sunset', 'dusk']

def find_sun(
    solar_dict: pd.Series,
) -> Dict[str, Any]:
    loc = LocationInfo(
        timezone=solar_dict.get("timezone"),
        latitude=solar_dict.get("latitude"),
        longitude=solar_dict.get("longitude"),
    )
    s = sun(
        loc.observer,
        solar_dict["date"],
        tzinfo=loc.tzinfo,
    )
    solar_dict = {
        "solar_id": str(uuid.uuid4()),
        "site_id": solar_dict["site_id"],
        "date": solar_dict["date"],
        **{
            k: s[k].replace(tzinfo=None)
            for k in s
        },
    }
    return solar_dict

def find_solar_boundaries(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df['dawn_end'] = df["sunrise"].add(df["sunrise"].subtract(df["dawn"]))
    df['dusk_start'] = df["sunset"].subtract(df["dusk"].subtract(df["sunset"]))
    return df

def find_relative_solar(
    df: pd.DataFrame,
) -> pd.DataFrame:
    relative_to_time_columns = [f'hours after {t}' for t in tod_cols]
    times = df["timestamp"].to_numpy().reshape(-1, 1)
    df[relative_to_time_columns] = (
        df[tod_cols].subtract(times, axis=0)
        .div(pd.Timedelta(hours=1))
        .mul(-1)
    )
    df['dddn'] = 'day'
    df['dddn'] = df['dddn'].mask(~df.timestamp.between(df.dawn, df.dusk), 'night')
    df['dddn'] = df['dddn'].mask(df.timestamp.between(df.dawn, df.sunrise + (df.sunrise - df.dawn)), 'dawn')
    df['dddn'] = df['dddn'].mask(df.timestamp.between(df.sunset - (df.dusk - df.sunset), df.dusk), 'dusk')
    return df

def find_date_and_hour(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df.loc[:, "hour"] = df.timestamp.dt.hour.astype(np.int8)
    df.loc[:, "date"] = df.timestamp.dt.date
    return df

def solartimes(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate solar event times (dawn, sunrise, sunset, dusk) for each timestamp in the given dataframe.

    Args:
        dataframe (pd.DataFrame): The input dataframe containing a unique file id, latitude, longitude and timestamp

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
    # Replace the location/recorder/date metadata
    df = df.join(df.apply(find_sun, axis=1, result_type='expand'))
    # Grouping into dawn/day/dusk/night
    df['dawn end'] = df.sunrise.add(df.sunrise.subtract(df.dawn))
    df['dusk start'] = df.sunset.subtract(df.dusk.subtract(df.sunset))

    # Convert to hours past event format
    relative_to_time_columns = [f'hours after {t}' for t in tod_cols]
    times = df.timestamp.to_numpy().reshape(-1, 1)
    df[relative_to_time_columns] = df[tod_cols].subtract(times, axis=0).div(pd.Timedelta(hours=1)).mul(-1)

    df['dddn'] = 'day'
    df['dddn'] = df['dddn'].mask(~df.timestamp.between(df.dawn, df.dusk), 'night')
    df['dddn'] = df['dddn'].mask(df.timestamp.between(df.dawn, df.sunrise + (df.sunrise - df.dawn)), 'dawn')
    df['dddn'] = df['dddn'].mask(df.timestamp.between(df.sunset - (df.dusk - df.sunset), df.dusk), 'dusk')

    # drop merge columns used to calculate solar information, preserve join columns for downstream join
    df = df.drop([col for col in locations.columns if col not in join_columns], axis=1)

    return df
