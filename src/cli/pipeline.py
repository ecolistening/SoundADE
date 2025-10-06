import argparse
import dask
import datetime as dt
import logging
import os
import pandas as pd
import shutil
import time

from dask import bag as db
from dask import dataframe as dd
from dask import config as cfg
from dask.distributed import Client
from pathlib import Path
from typing import Any, Tuple

from soundade.hpc.arguments import DaskArgumentParser

from cli.index_sites import index_sites
from cli.index_audio import index_audio
from cli.index_solar import index_solar
from cli.index_weather import index_weather
from cli.acoustic_features import acoustic_features
from cli.birdnet_detections import birdnet_detections

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def pipeline(
    root_dir: Path,
    config_path: Path,
    save_dir: Path,
    sitesfile: Path | None,
    high_pass_filter: int,
    dc_correction: int,
    partition_size: int = None,
    npartitions: int = None,
    no_birdnet: bool = False,
    no_indices: bool = False,
    **kwargs: Any,
) -> None:
    # setup data sinks
    save_dir = save_dir.expanduser()
    save_dir.mkdir(exist_ok=True, parents=True)
    sites_path = sitesfile or save_dir / "locations_table.parquet"
    tmp_files_path = save_dir / "files_table.tmp.parquet"
    files_path = save_dir / "files_table.parquet"
    solar_path = save_dir / "solar_table.parquet"
    recording_acoustic_features_path = save_dir / "recording_acoustic_features_table.parquet"
    birdnet_species_probs_path = save_dir / "birdnet_species_probs_table.parquet"
    # begin timing
    start_time = time.time()
    # index sites
    log.info(f"Processing site information")
    sites_df = index_sites(
        root_dir=root_dir,
        config_path=config_path,
        out_file=sites_path,
    )
    # index files
    log.info(f"Indexing audio files")
    files_df, _ = index_audio(
        root_dir=root_dir,
        config_path=config_path,
        out_file=tmp_files_path,
        sites_ddf=dd.from_pandas(sites_df),
        compute=True,
    )
    # index site-specific information
    log.info(f"Indexing solar times")
    index_solar(
        files_ddf=dd.from_pandas(files_df),
        sites_ddf=dd.from_pandas(sites_df),
        infile=files_path,
        outfile=solar_path,
        compute=True,
    )
    # cleanup old files table
    shutil.rmtree(tmp_files_path)
    log.info(f"Indexing weather data")
    index_weather(
        files_df=files_df,
        sites_df=sites_df,
        save_dir=save_dir,
    )
    # extract acoustic featres
    futures = []
    if not no_indices:
        log.info(f"Extracting acoustic features")
        acoustic_features_ddf, acoustic_features_future = acoustic_features(
            root_dir=root_dir,
            config_path=config_path,
            files_df=files_df,
            outfile=recording_acoustic_features_path,
            high_pass_filter=high_pass_filter,
            dc_correction=dc_correction,
            compute=False,
        )
        futures.append(acoustic_features_future)
    # extract birdnet species scores
    if not no_birdnet:
        log.info(f"Extracting BirdNET species probabilities")
        birdnet_species_ddf, birdnet_species_future = birdnet_detections(
            root_dir=root_dir,
            config_path=config_path,
            files_df=files_df,
            sites_df=sites_df,
            outfile=birdnet_species_probs_path,
            compute=False,
        )
        futures.append(birdnet_species_future)
    # compute the graph
    log.info(f"Processing...")
    dask.compute(*futures)
    # and we're done!
    log.info("Pipeline complete")
    log.info(f"Time taken: {str(dt.timedelta(seconds=time.time() - start_time))}")

def main(
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

    pipeline(**kwargs)

def get_base_parser():
    parser = DaskArgumentParser(
        description="Run the full pipeline",
        add_help=False,
    )
    parser.add_argument(
        "--root-dir",
        type=lambda p: Path(p).expanduser(),
        help="Root directory containing audio files. Defaults to /data for container builds.",
    )
    parser.add_argument(
        '--config-path',
        type=lambda p: Path(p).expanduser(),
        help='/path/to/dataset/config.yaml',
    )
    parser.add_argument(
        '--save-dir',
        type=lambda p: Path(p).expanduser(),
        help="Target directory for results. Defaults to /results for container builds.",
    )
    parser.add_argument(
        '--sitesfile',
        type=lambda p: Path(p).expanduser(),
        help="Path to a parquet file with columns ('site_id', 'site_name',  'latitude',  'longitude',  'timezone')",
    )
    parser.add_argument(
        "--dc-correction",
        type=int,
        help="Set to 1 to apply DC Correction by subtracting the mean",
    )
    parser.add_argument(
        "--high-pass-filter",
        type=int,
        help="Set to 1 to apply a high pass filter",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        help="Threads per worker",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Sets single-threaded for debugging.",
    )
    parser.add_argument(
        "--no-birdnet",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--no-indices",
        default=False,
        action="store_true",
    )
    parser.set_defaults(func=main, **{
        "root_dir": "/data",
        "config_path": "/config.yml",
        "save_dir": "/results",
        'local': os.environ.get("LOCAL", True),
        "memory": os.environ.get("MEM_PER_CPU", 0),
        "cores": os.environ.get("CORES", 0),
        "threads_per_worker": os.environ.get("THREADS_PER_WORKER", 1),
        "dc_correction": os.environ.get("DC_CORR", 0),
        "high_pass_filter": os.environ.get("HIGH_PASS_FILTER", 1),
    })
    return parser

def register_subparser(subparsers):
    parser = subparsers.add_parser(
        "pipeline",
        help="Process a full audio directory",
        parents=[get_base_parser()],
        add_help=True,
    )

if __name__ == "__main__":
    parser = get_base_parser()
    args = parser.parse_args()
    log.info(args)
    args.func(**vars(args))
