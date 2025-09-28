import argparse
import datetime as dt
import itertools
import logging
import openmeteo_requests
import os
import pandas as pd
import requests_cache
import time

from pathlib import Path
from retry_requests import retry
from requests import Session
from typing import Any, Dict, Tuple, List
from urllib.error import HTTPError

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

_client = None

def init_client(*args, **kwargs):
    global _client
    if _client is None:
        _client = OpenMeteoClient(*args, **kwargs)
    return _client

def OpenMeteoClient(
    retries: int = 5,
    expire_after: int = -1,
    backoff_factor: float = 0.2,
) -> openmeteo_requests.Client:
    session = retry(Session(), retries=retries, backoff_factor=backoff_factor)
    return openmeteo_requests.Client(session=session)

DEFAULT_WEATHER_COLUMNS = {
    "temperature_2m": "Temperature at 2m (°C)",
    "precipitation": "Precipitation (cm)",
    "rain": "Rain (cm)",
    "snowfall": "Snowfall (cm)",
    "wind_speed_10m": "Wind Speed at 10m (km/h)",
    "wind_speed_100m": "Wind Speed at 100m (km/h)",
    "wind_direction_10m": "Wind Direction at 10m (km/h)",
    "wind_direction_100m": "Wind Direction at 100m (km/h)",
    "wind_gusts_10m": "Wind Gusts at 10m (km/h)",
}

def find_weather_hourly(
    data_dict: Dict[str, Any],
    columns: List[str] = DEFAULT_WEATHER_COLUMNS.keys(),
) -> Dict[str, Any]:
    """
    Requires: dictionary of the form:
    {
        site_id: str | int,
        latitude: float,
        longitude: float,
        start_date: datetime64[us],
        end_date: datetime64[us],
    }

    Returns: a list of dictionaries of the form:
    {
        site_id: str | int,
        timestamp: datetime64[us],
        **WEATHER_COLUMNS
    }
    """
    params = {
        "latitude": data_dict["latitude"],
        "longitude": data_dict["longitude"],
        "start_date": data_dict["start_date"],
        "end_date": data_dict["end_date"],
        "hourly": columns,
    }

    client = init_client()
    try:
        responses = client.weather_api(OPEN_METEO_ARCHIVE_URL, params=params)
        hourly = responses[0].Hourly()
        data = {
            column: hourly.Variables(i).ValuesAsNumpy()
            for i, column in enumerate(columns)
        }
        df = pd.DataFrame({
            "timestamp": pd.date_range(
                pd.to_datetime(hourly.Time(), unit="s"),
                pd.to_datetime(hourly.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ),
            **data,
        })
        df["site_id"] = data_dict["site_id"]
        return df.to_dict(orient="records")
    except HTTPError as e:
        log.warning("Request to OpenMeteo failed, are you connected to the internet?")
        log.error(e)
        return []
    except Exception as e:
        log.warning("Failed to fetch weather data")
        log.error(e)
        return []

def index_weather(
    files_df: pd.DataFrame,
    sites_df: pd.DataFrame,
    save_dir: Path,
    **kwargs: Any,
) -> None:
    if sites_df.index.name == "site_id":
        sites_df = sites_df.reset_index()
    log.info("Fetching weather data from open meteo")
    df = (
        files_df[["site_id", "timestamp"]]
        .groupby("site_id")
        .agg(start_date=("timestamp", lambda x: x.dt.date.min()), end_date=("timestamp", lambda x: x.dt.date.max()))
        .merge(sites_df[["site_id", "latitude", "longitude"]], on="site_id", how="left")
        .reset_index()
    )
    weather_df = pd.DataFrame(list(itertools.chain(*list(map(find_weather_hourly, df.to_dict(orient="records"))))))
    weather_df.to_parquet(save_dir / "weather_table.parquet")
    log.info(f"Weather data persisted to {save_dir / 'weather_table.parquet'}")
    return weather_df

def main(
    files_path: Path,
    sites_path: Path,
    save_dir: Path | None,
    **kwargs: Any,
) -> None:
    start_time = time.time()
    index_weather(
        files_df=pd.read_parquet(files_path),
        sites_df=pd.read_parquet(sites_path),
        save_dir=save_dir or files_path.parent,
        **kwargs,
    )
    log.info(f"Weather index complete")
    log.info(f"Time taken: {str(dt.timedelta(seconds=time.time() - start_time))}")

def get_base_parser():
    parser = argparse.ArgumentParser(
        description='Fetch and persist weather data for audio files',
        add_help=False,
    )
    parser.add_argument(
        "--files-path",
        required=True,
        type=lambda p: Path(p).expanduser(),
        help="/path/to/files_table.parquet"
    )
    parser.add_argument(
        "--sites-path",
        required=True,
        type=lambda p: Path(p).expanduser(),
        help="/path/to/locations_table.parquet"
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        type=lambda p: Path(p).expanduser(),
        help="/path/to/save/dir"
    )
    parser.set_defaults(func=main, **{
        "files_path": "/results/files_table.parquet",
        "sites_path": "/results/locations_table.parquet",
        "save_dir": "/results",
    })
    return parser

def register_subparser(subparsers):
    parser = subparsers.add_parser(
        "index_weather",
        help="Fetch weather data for duration of recording period",
        parents=[get_base_parser()],
        add_help=True,
    )

if __name__ == '__main__':
    parser = get_base_parser()
    args = parser.parse_args()
    log.info(args)
    args.func(**vars(args))
