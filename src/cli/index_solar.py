import os
import datetime as dt
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import time

from dask import config as cfg
from dask import dataframe as dd
from dask.distributed import Client
from pathlib import Path
from typing import Any, Tuple

from soundade.data.solar import (
    tod_cols,
    find_sun,
    find_date,
    find_solar_boundaries,
    find_relative_solar,
    find_dddn,
)
from soundade.hpc.arguments import DaskArgumentParser
from soundade.hpc.cluster import clusters

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

cfg.set({
    "distributed.scheduler.worker-ttl": None
})

def index_solar(
    files_ddf: dd.DataFrame,
    sites_ddf: dd.DataFrame,
    infile: str | Path,
    outfile: str | Path,
    partition_size: int = None,
    npartitions: int = None,
    compute: bool = True,
) -> Tuple[dd.DataFrame, dd.DataFrame, dd.Scalar] | Tuple[pd.DataFrame, pd.DataFrame]:
    # extract date information and drop timestamp
    log.info("Extracting date from timestamp")
    files_ddf = (
        files_ddf
        .map_partitions(find_date)
    )

    # stage 1: extract unique solar table, referencing
    log.info("Building solar dataframe containing dawn/dusk/sunrise/sunset information")

    ddf = (
        files_ddf[["site_id", "date"]]
        # drop duplicates at a site on a date
        .drop_duplicates(subset=["site_id", "date"], keep="last")
        # join location information on site_id for all solar info
        .join(sites_ddf, on="site_id", how="left")
        .persist()
    )
    b = (
        ddf
        # map to a bag since the operation is not vectorised
        .to_bag(format="dict")
    )
    b = b.map(find_sun)
    # store in a dataframe
    solar_ddf = (
        b.to_dataframe()
        .map_partitions(find_solar_boundaries)
        .astype({"site_id": "string[pyarrow]"})
    )

    log.info("Merging with file index, appending times relative to solar timestamps")

    # stage 2: join solar info with the file table and calculate file-specific solar data
    solar_columns = ["date", *tod_cols]
    site_columns = ["timezone", "latitude", "longitude"]
    files_ddf = (
        files_ddf
        # join on location and date
        .merge(solar_ddf[["site_id", *site_columns, *solar_columns]], on=["site_id", "date"], how="left")
        # extract dddn
        .map_partitions(find_dddn)
        # extract solar info relative to time
        .map_partitions(find_relative_solar)
        # drop from files table
        .drop(solar_columns, axis=1)
        .drop(site_columns, axis=1)
    )

    if compute:
        solar_df = solar_ddf.compute()
        solar_df.to_parquet(Path(outfile), index=False)
        files_df = files_ddf.compute()
        files_df.to_parquet(Path(infile), index=False)
        log.info(f"Solar index saved to {outfile}")
        log.info(f"File index updated at {infile}")
        return files_df, solar_df

    files_future = files_ddf.to_parquet(
        Path(infile),
        version='2.6',
        allow_truncated_timestamps=True,
        write_index=False,
        compute=False,
    )
    solar_future = solar_ddf.to_parquet(
        Path(outfile),
        version='2.6',
        allow_truncated_timestamps=True,
        write_index=False,
        compute=False,
    )

    log.info(f"Solar queued, will persist to {outfile}")
    log.info(f"Indexing files queued, will overwrite {infile}")

    return files_ddf, solar_ddf, files_future, solar_future

def main(
    infile: str | Path,
    outfile: str | Path,
    sitesfile: str | Path,
    memory: int = 0,
    cores: int = 0,
    jobs: int = 0,
    npartitions: int | None = None,
    local: bool = True,
    threads_per_worker: int = 1,
    compute: bool = False,
    debug: bool = False,
    **kwargs
) -> None:
    """
    Process and append metadata to the input data.

    Args:
        infile (str, required): Path to the input data file.
        outfile (str, required): Path to the output file where the processed data will be saved.
        sitesfile: (str, required): Path to the sites table. Uses the join key 'site_id' to attach location information.
        memory (int, optional): Amount of memory (in GB) to allocate for the computation.
        cores (int, optional): Number of CPU cores to use for the computation.
        jobs (int, optional): Number of parallel jobs to run.
        local (bool, optional): Flag indicating whether to use a local cluster for computation.
        compute (bool, optional): Flag indicating whether to persist parquet eagerly. Defaults to false.
        debug (bool, optional): Flag indicating whether to run synchronously. Defaults to false.

    Returns:
        dd.DataFrame
    """
    assert infile is not None
    assert outfile is not None

    if not local:
        cluster = clusters[cluster](
            cores=cores,
            memory=memory,
            queue=queue,
            name=None
        )
        log.info(cluster.job_script())
        cluster.scale(jobs=jobs)
        client = Client(cluster)
    else:
        if debug:
            cfg.set(scheduler='synchronous')

        client = Client(
            n_workers=cores,
            threads_per_worker=threads_per_worker,
            memory_limit=f'{memory}GiB'
        )
        log.info(client)

    start_time = time.time()

    index_solar(
        files_ddf=dd.read_parquet(infile),
        sites_ddf=dd.read_parquet(sitesfile),
        infile=infile,
        outfile=outfile,
        npartitions=npartitions,
        compute=compute,
    )

    log.info(f"Solar index complete")
    log.info(f"Time taken: {time.time() - start_time}")

def get_base_parser():
    parser = DaskArgumentParser(
        description='Extract and merge solar times for audio files',
        add_help=False,
    )
    parser.add_argument(
        "--sitesfile",
        default=None,
        help="Parquet file containing site information."
    )
    parser.add_argument(
        '--save-preprocessed',
        default=None,
        help='Save the preprocessed files to directory.'
    )
    parser.add_argument(
        '--compute',
        default=False,
        action='store_true',
        help='Aggregate the dataframe in memory before saving to parquet.'
    )
    parser.add_argument(
        '--debug',
        default=False,
        action='store_true',
        help='Sets single-threaded for debugging.'
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="Threads per worker",
    )
    parser.set_defaults(func=main, **{
        "infile": "/".join([os.environ.get("DATA_PATH", "/data"), "files_table.parquet"]),
        "outfile": "/".join([os.environ.get("DATA_PATH", "/data"), "solar_table.parquet"]),
        "sitesfile": "/".join([os.environ.get("DATA_PATH", "/data"), "locations_table.parquet"]),
        "memory": os.environ.get("MEM_PER_CPU", 0),
        "cores": os.environ.get("CORES", 1),
        "local": True,
        "threads_per_worker": 1,
    })
    return parser

def register_subparser(subparsers):
    parser = subparsers.add_parser(
        "index_solar",
        help="Extract and merge solar times for audio files",
        parents=[get_base_parser()],
        add_help=True,
    )

if __name__ == '__main__':
    parser = get_base_parser()
    args = parser.parse_args()
    log.info(args)
    args.func(**vars(args))
