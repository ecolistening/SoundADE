import logging
import pandas as pd
import numpy as np
import uuid

from astral import LocationInfo
from astral import sun
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

tod_cols = ['dawn', 'sunrise', 'noon', 'sunset', 'dusk']

def find_sun(
    data_dict: pd.Series,
) -> Dict[str, Any]:
    loc = LocationInfo(
        timezone=data_dict.get("timezone"),
        latitude=data_dict.get("latitude"),
        longitude=data_dict.get("longitude"),
    )
    solar_dict = {}
    try:
        solar_dict["dawn"] = sun.dawn(loc.observer, data_dict["date"], tzinfo=loc.tzinfo).replace(tzinfo=None)
    except ValueError as e:
        log.warning(e)
        solar_dict["dawn"] = None
    try:
        solar_dict["sunrise"] = sun.sunrise(loc.observer, data_dict["date"], tzinfo=loc.tzinfo).replace(tzinfo=None)
    except ValueError as e:
        log.warning(e)
        solar_dict["sunrise"] = None
    try:
        solar_dict["noon"] = sun.noon(loc.observer, data_dict["date"], tzinfo=loc.tzinfo).replace(tzinfo=None)
    except ValueError as e:
        log.warning(e)
        solar_dict["noon"] = None
    try:
        solar_dict["sunset"] = sun.sunset(loc.observer, data_dict["date"], tzinfo=loc.tzinfo).replace(tzinfo=None)
    except ValueError as e:
        log.warning(e)
        solar_dict["sunset"] = None
    try:
        solar_dict["dusk"] = sun.dusk(loc.observer, data_dict["date"], tzinfo=loc.tzinfo).replace(tzinfo=None)
    except ValueError as e:
        log.warning(e)
        solar_dict["dusk"] = None
    return {
        **data_dict,
        **solar_dict,
    }

def find_solar_boundaries(
    df: pd.DataFrame,
) -> pd.DataFrame:
    try:
        df['dawn_end'] = df["sunrise"].add(df["sunrise"].subtract(df["dawn"]))
        df['dusk_start'] = df["sunset"].subtract(df["dusk"].subtract(df["sunset"]))
    except ValueError as e:
        df["dawn_end"] = None
        df["dusk_start"] = None
    return df

def classify_phase(row, twilight_angle=-6):
    loc = LocationInfo(
        timezone=row.timezone,
        latitude=row.latitude,
        longitude=row.longitude,
    )
    elevation = sun.elevation(loc.observer, row.timestamp)
    azimuth = sun.azimuth(loc.observer, row.timestamp)
    dddn = None
    if elevation > 0:
        dddn = "day"
    elif elevation <= twilight_angle:
        dddn = "night"
    elif azimuth < 180:
        dddn = "dawn"
    else:
        dddn = "dusk"
    return dddn

def find_dddn(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["dddn"] = df.apply(classify_phase, axis=1)
    return df

# DEPRECATED: left for future reference on our previous method
# def find_dddn(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     df['dddn'] = 'day'
#     df['dddn'] = df['dddn'].mask(~df.timestamp.between(df.dawn, df.dusk), 'night')
#     df['dddn'] = df['dddn'].mask(df.timestamp.between(df.dawn, df.sunrise + (df.sunrise - df.dawn)), 'dawn')
#     df['dddn'] = df['dddn'].mask(df.timestamp.between(df.sunset - (df.dusk - df.sunset), df.dusk), 'dusk')
#     return df

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
    return df

def find_date(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df.loc[:, "date"] = df.timestamp.dt.date.astype("datetime64[ns]")
    return df
