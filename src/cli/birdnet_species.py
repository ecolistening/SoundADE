import argparse
import os
import time
import pathlib
import logging
import pandas as pd

from dask import config as cfg
from dask import dataframe as dd
from dask.distributed import Client
from pathlib import Path
from typing import Any, Tuple

from soundade.audio.birdnet import species_probs, species_probs_meta, species_probs_as_df
from soundade.hpc.arguments import DaskArgumentParser

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

cfg.set({
    "distributed.scheduler.worker-ttl": None
})

def birdnet_species(
    files: dd.DataFrame,
    sites: pd.DataFrame | dd.DataFrame,
    outfile: str | Path,
    min_conf: float,
    npartitions: int = None,
    compute: bool = False,
) -> Tuple[dd.DataFrame, dd.Scalar] | pd.DataFrame:
    start_time = time.time()

    log.info(f"Extracting BirdNET species probabilities...")
    ddf = (
        files[["file_cid", "local_file_path", "timestamp", "site_id", "valid"]]
        .merge(sites[["site_id", "latitude", "longitude"]], on="site_id", how="left")
        # .map_partitions(species_probs_as_df, min_conf=min_conf, meta=species_probs_meta())
        .to_bag(format="dict")
        .filter(lambda file_dict: file_dict["valid"])
        .map(species_probs, min_conf=min_conf)
        .filter(len)
        .flatten()
        .to_dataframe(meta=species_probs_meta())
    )

    if compute:
        df = ddf.compute()
        df.to_parquet(Path(outfile), index=False)
        log.info(f"Time taken: {time.time() - start_time}")
        log.info(f"BirdNET species detections saved to {outfile}")
        return df

    future = ddf.to_parquet(
        Path(outfile),
        version='2.6',
        allow_truncated_timestamps=True,
        write_index=False,
        compute=False
    )

    log.info(f"BirdNET processing queued, will persist to {outfile}")

    return ddf, future

def main(
    cluster: str | None = None,
    infile: str | Path | None = None,
    outfile: str | Path | None = None,
    sitesfile: str | None = None,
    memory: int = 0,
    cores: int = 1,
    jobs: int = 1,
    queue: str = "general",
    min_conf: float = 0.5,
    npartitions: int | None = None,
    local: bool = True,
    threads_per_worker: int = 1,
    compute: bool = False,
    debug=False,
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
        min_conf (float, optional): Confidence threshold used for omitting low confidence predictions

    Returns:
        None

    Examples:
        >>> main(infile='./audio_file_index.parquet', outfile='./birdnet_predictions.parquet', sitesfile='./locations.parquet')
    """
    assert infile is not None
    assert outfile is not None

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

    birdnet_species(
        files=dd.read_parquet(infile),
        sites=dd.read_parquet(sitesfile),
        outfile=outfile,
        min_conf=min_conf,
        npartitions=npartitions,
        compute=compute,
    )

def get_base_parser():
    parser = DaskArgumentParser(
        description="Extract species probabilities using BirdNET",
        add_help=False,
    )
    parser.add_argument(
        "--sitesfile",
        default=None,
        help="Parquet file containing site information."
    )
    parser.add_argument(
        "--min-conf",
        required=False,
        type=float,
        default=0.5,
        help="BirdNET confidence threshold"
    )
    parser.add_argument(
        '--compute',
        default=False,
        action='store_true',
        help='Aggregate the dataframe in memory before saving to parquet.'
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Sets single-threaded for debugging."
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="Threads per worker",
    )
    parser.set_defaults(func=main, **{
        "infile": "/data/files_table.parquet",
        "outfile": "/data/birdnet_species_probs.parquet",
        "sitesfile": "./data/locations.parquet",
        "min_conf": 0.3,
        "memory": 0,
        "cores": 1,
        "local": True,
        "threads_per_worker": 1,
    })
    return parser

def register_subparser(subparsers):
    parser = subparsers.add_parser(
        "birdnet_species",
        help="Extract species probabilities using BirdNET",
        parents=[get_base_parser()],
        add_help=True,
    )

if __name__ == "__main__":
    parser = get_base_parser()
    args = parser.parse_args()
    log.info(args)
    args.func(**vars(args))
