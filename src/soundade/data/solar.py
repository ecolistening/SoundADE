from pathlib import Path

import pandas as pd
from astral import LocationInfo
from astral.sun import sun

<<<<<<< Updated upstream
=======
from typing import Union

>>>>>>> Stashed changes
locations_default = Path(__file__).parent / '../../../data/site_locations.parquet'

def find_sun(r, locations):
    loc = LocationInfo(name=r.location,
                       region=locations[locations.location == r.location].iloc[0, :].country,
                       timezone=locations[locations.location == r.location].iloc[0, :].timezone,
                       latitude=locations[(locations.location == r.location) & (locations.recorder == r.recorder)].latitude,
                       longitude=locations[(locations.location == r.location) & (locations.recorder == r.recorder)].longitude
                       )
    s = sun(loc.observer, r.date, tzinfo=loc.tzinfo)

    return dict([(k, s[k].replace(tzinfo=None)) for k in s])

tod_cols = ['dawn', 'sunrise', 'noon', 'sunset', 'dusk']

def solartimes(dataframe: pd.DataFrame, locations: Union[pd.DataFrame, Path, str] = locations_default) -> pd.DataFrame:
    """
    Calculate solar event times (dawn, sunrise, sunset, dusk) for each timestamp in the given dataframe.

    Args:
        dataframe (pd.DataFrame): The input dataframe containing timestamps and other data.
        locations (pd.DataFrame | Path | str, optional): The locations dataframe or path to the locations file. Defaults to locations_default.

    Returns:
        pd.DataFrame: The dataframe with solar times and other calculated features.

    Raises:
        FileNotFoundError: If the locations file is not found.

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
    # print(f'dataframe: {dataframe.shape}')

    # There may be duplicate indices in the dataframe, so we reset the indices,
    # creating a column called 'index' that can be used to join metadata and features later
    dataframe = dataframe.reset_index()

    # Split the dataframe into metadata and features columns
    metadata = dataframe.iloc[:,:dataframe.columns.get_loc('0')]
    features = dataframe.iloc[:,dataframe.columns.get_loc('0'):]

    if len(dataframe.index) == 0:
        d = {t:'M' for t in tod_cols} | {f'hours before {t}':'float64' for t in tod_cols} | {'dawn start': 'M', 'dusk_start': 'M', 'dddn':'string'}
        return pd.concat([metadata,pd.DataFrame(columns=list(d.keys())).astype(d),features])

    # Load locations, if necessary
    if isinstance(locations, Path) or isinstance(locations, str):
        locations = pd.read_parquet(locations)

    #TODO: Why do we do this?
    df_meta = dataframe.assign(date=lambda r: r.timestamp.dt.date)[['location', 'recorder', 'date']].drop_duplicates()

    suntimes = df_meta.apply(find_sun, locations=locations, axis=1, result_type='expand')
    
    # Remove TZ info because our dates in the DB are without time zones
    for c in suntimes.columns:
        suntimes[c] = suntimes[c].dt.tz_localize(None)

    # Replace the location/recorder/date metadata
    suntimes = df_meta.join(suntimes)

    # print(f'Suntimes: {suntimes.shape}')

    # We reset_index again here to preserve the new index in the merge.
    # This creates a column called 'level_0' that is set as the index after the merge operation.
    df_suntimes = metadata.reset_index().assign(date=lambda r: r.timestamp.dt.date).merge(suntimes, on=['location', 'recorder', 'date'], how='inner').drop(columns=['date']).set_index('level_0')

    # display(df_suntimes)

    # print(f'df suntimes: {df_suntimes.shape}')

    # Convert to hours past event format
    df_suntimes[[f'hours after {t}' for t in tod_cols]] = df_suntimes[tod_cols].subtract(
        df_suntimes.timestamp.to_numpy().reshape(-1, 1), axis=0).div(pd.Timedelta(hours=1)).mul(-1)

    # Grouping into dawn/day/dusk/night
    df_suntimes['dawn end'] = df_suntimes.sunrise.add(df_suntimes.sunrise.subtract(df_suntimes.dawn))
    df_suntimes['dusk start'] = df_suntimes.sunset.subtract(df_suntimes.dusk.subtract(df_suntimes.sunset))

    df_suntimes['dddn'] = 'day'
    df_suntimes['dddn'] = df_suntimes['dddn'].mask(
        ~df_suntimes.timestamp.between(df_suntimes.dawn, df_suntimes.dusk), 'night'
    )
    df_suntimes['dddn'] = df_suntimes['dddn'].mask(
        df_suntimes.timestamp.between(df_suntimes.dawn, df_suntimes.sunrise + (df_suntimes.sunrise - df_suntimes.dawn)),
        'dawn'
    )
    df_suntimes['dddn'] = df_suntimes['dddn'].mask(df_suntimes.timestamp.between(
        df_suntimes.sunset - (df_suntimes.dusk - df_suntimes.sunset), df_suntimes.dusk), 'dusk'
    )

    # print(f'join: {df_suntimes.join(features).shape}')

    # The original index is reset as map_partitions assumes that the index does not change.
    df = df_suntimes.join(features).set_index('index')

    return df