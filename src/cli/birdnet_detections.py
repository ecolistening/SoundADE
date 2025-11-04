import argparse
import dask
import datetime as dt
import logging
import os
import pandas as pd
import shutil
import time

from dask import config as cfg
from dask import bag as db
from dask import dataframe as dd
from dask.distributed import Client
from pathlib import Path
from typing import Any, Tuple

from soundade.data.dataset import Dataset
from soundade.audio.birdnet import species_probs, species_probs_meta
from soundade.hpc.arguments import DaskArgumentParser

PYARROW_VERSION = "2.6"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def birdnet_detections(
    root_dir: Path,
    config_path: Path,
    files_df: pd.DataFrame,
    sites_df: pd.DataFrame,
    outfile: str | Path,
    npartitions: int = None,
    compute: bool = False,
    **kwargs: Any,
) -> Tuple[dd.DataFrame, dd.Scalar] | pd.DataFrame:
    if sites_df is not None and sites_df.index.name == "site_id":
        sites_df = sites_df.reset_index()

    root_dir = Path(root_dir).expanduser()
    dataset = Dataset.from_config_path(config_path)

    log.info(f"Setting up BirdNET species detection extraction pipeline.")
    log.info("Corrupt files will be filtered.")

    files_df = files_df[files_df.valid]
    files_df["local_file_path"] = root_dir / files_df["file_path"].astype(str)
    columns = ["file_id", "local_file_path", "timestamp", "site_id"]

    if sites_df is not None:
        df = files_df[columns].merge(
            sites_df[["site_id", "latitude", "longitude"]],
            on="site_id"
        )
    else:
        df = files_df

    records = df.to_dict(orient="records")
    b = db.from_sequence(records, npartitions=npartitions)

    log.info(f"Partitions after load: {b.npartitions}")
    params = dataset.birdnet_params
    log.info(f"Extracting detection probabilities with {params=} for {len(files_df)} files")

    ddf = (
        b.map(species_probs, **params)
        .filter(len)
        .flatten()
        .to_dataframe(meta=species_probs_meta())
    )

    future = ddf.to_parquet(
        Path(outfile),
        version=PYARROW_VERSION,
        allow_truncated_timestamps=True,
        write_index=False,
        compute=False
    )

    log.info(f"Queued BirdNET detections, will persist to {outfile}")

    if compute:
        dask.compute(future)
        return pd.read_parquet(Path(outfile)), None

    return ddf, future

def main(
    infile: Path,
    outfile: str | Path,
    sitesfile: Path,
    config_path: Path,
    cluster: str | None,
    memory: int,
    cores: int,
    jobs: int,
    queue: str,
    local: bool,
    threads_per_worker: int,
    debug: bool,
    **kwargs: Any,
) -> None:
    """
    Extract species detections in audio files indexed by infile using BirdNET. Not that this only persists species _detections_,
    and therefore does not contain references to files where no species were detected, which are dropped from the results.

    Args:
        cluster (str, optional): Name of the cluster to use. 'arc' or 'altair' or None if local==True. Defaults to None.
        infile (str): Path to the input data file.
        outfile (str): Path to the output file where the processed data will be saved.
        sitesfile (str): Path to sites file including lat/lng information and the relevant join key, e.g. location/site
        memory (int, optional): Memory limit for each worker in GB. Defaults to 32.
        cores (int, optional): Number of CPU cores per worker. Defaults to 8.
        jobs (int, optional): Number of worker jobs to start. Defaults to 12.
        npartitions (int): Number of partitions to repartition the data into.

    Returns:
        None

    Examples:
        >>> main(infile='./audio_file_index.parquet', outfile='./birdnet_predictions.parquet', sitesfile='./locations.parquet')
    """
    if not local:
        Cluster = clusters[cluster]
        cluster = Cluster(
            cores=cores,
            memory=memory,
            queue=queue,
            name=None,
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
            memory_limit=f'{memory}GiB',
        )
        log.info(client)

    start_time = time.time()

    _, future = birdnet_detections(
        files_df=pd.read_parquet(infile),
        sites_df=pd.read_parquet(sitesfile),
        outfile=outfile,
        config_path=config_path,
        **kwargs,
    )
    dask.compute(future)

    shutil.copy(config_path, outfile.parent / "config.yaml")

    log.info(f"BirdNET detection extraction complete")
    log.info(f"Time taken: {str(dt.timedelta(seconds=time.time() - start_time))}")

    client.close()

def get_base_parser():
    parser = DaskArgumentParser(
        description="Extract species probabilities using BirdNET",
        add_help=False,
    )
    parser.add_argument(
        "--root-dir",
        type=lambda p: Path(p).expanduser(),
        help="Root directory of the audio files (nested folder structure permitted)",
    )
    parser.add_argument(
        '--config-path',
        type=lambda p: Path(p).expanduser(),
        help='/path/to/dataset/config.yaml',
    )
    parser.add_argument(
        "--sitesfile",
        type=lambda p: Path(p).expanduser(),
        default=None,
        help="Parquet file containing site information.",
    )
    parser.add_argument(
        "--min-conf",
        required=False,
        type=float,
        default=0.5,
        help="BirdNET confidence threshold",
    )
    parser.add_argument(
        '--compute',
        default=False,
        action='store_true',
        help='Aggregate the dataframe in memory before saving to parquet.',
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Sets single-threaded for debugging.",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="Threads per worker",
    )
    parser.set_defaults(func=main, **{
        "root_dir": "/data",
        "config_path": "/config.yaml",
        "infile": "/results/files_table.parquet",
        "outfile": "/results/birdnet_species_probs_table.parquet",
        "sitesfile": "/results/locations_table.parquet",
        "memory": os.environ.get("MEM_PER_CPU", 0),
        "cores": os.environ.get("CORES", 1),
        "local": True,
        "threads_per_worker": 1,
    })
    return parser

def register_subparser(subparsers):
    parser = subparsers.add_parser(
        "birdnet_detections",
        help="Extract species detection probabilities using BirdNET",
        parents=[get_base_parser()],
        add_help=True,
    )

if __name__ == "__main__":
    parser = get_base_parser()
    args = parser.parse_args()
    log.info(args)
    args.func(**vars(args))
