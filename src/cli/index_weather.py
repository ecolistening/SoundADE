import os
import argparse
import datetime as dt
import numpy as np
import pandas as pd
import pyarrow as pa
import time
import itertools
import logging
import openmeteo_requests
import requests_cache

from pathlib import Path
from typing import Any, Dict, Tuple, List
from retry_requests import retry

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

_client = None

def init_client(**kwargs):
    global _client
    if _client is None:
        _client = OpenMeteoClient(**kwargs)
    return _client

def OpenMeteoClient(
    retries: int = 5,
    expire_after: int = -1,
    backoff_factor: float = 0.2,
) -> openmeteo_requests.Client:
    cache_session = requests_cache.CachedSession('.cache', expire_after=expire_after)
    retry_session = retry(cache_session, retries=retries, backoff_factor=backoff_factor)
    return openmeteo_requests.Client(session=retry_session)

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
    d: Dict[str, Any],
    columns: List[str] = DEFAULT_WEATHER_COLUMNS.keys(),
) -> Dict[str, Any]:
    client = init_client()
    params = {
        "latitude": d["latitude"],
        "longitude": d["longitude"],
        "start_date": d["start_date"],
        "end_date": d["end_date"],
        "hourly": columns,
    }
    # Probably need to try/catch?
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
    df["site_id"] = d["site_id"]
    return df.to_dict(orient="records")

def index_weather(
    files_df: pd.DataFrame,
    sites_df: pd.DataFrame,
    save_dir: Path,
    **kwargs: Any,
) -> None:
    start_time = time.time()
    log.info("Fetching weather data from open meteo")
    df = (
        files_df[["site_id", "timestamp"]]
        .groupby("site_id")
        .agg(start_date=("timestamp", lambda x: x.dt.date.min()), end_date=("timestamp", lambda x: x.dt.date.max()))
        .join(sites_df[["latitude", "longitude", "site_name"]], how="left")
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
    index_weather(
        files_df=pd.read_parquet(files_path),
        sites_df=pd.read_parquet(sites_path),
        save_dir=save_dir or files_path.parent,
        **kwargs,
    )

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
        "files_path": "/".join([os.environ.get("DATA_PATH", "/data"), "files_table.parquet"]),
        "sites_path": "/".join([os.environ.get("DATA_PATH", "/data"), "locations_table.parquet"]),
        "save_dir": os.environ.get("DATA_PATH", "/data"),
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
