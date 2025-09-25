import argparse
import dask
import datetime as dt
import logging
import os
import pandas as pd
import time

from dask import bag as db
from dask import dataframe as dd
from dask import config as cfg
from dask.distributed import Client
from pathlib import Path
from typing import Any, Tuple

from soundade.hpc.arguments import DaskArgumentParser
from soundade.datasets import datasets

from cli.index_sites import index_sites
from cli.index_audio import index_audio
from cli.index_solar import index_solar
from cli.index_weather import index_weather
from cli.acoustic_features import acoustic_features
from cli.birdnet_detections import birdnet_detections

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def pipeline(
    root_dir: str | Path,
    save_dir: str | Path,
    sitesfile: str | Path | None,
    dataset: str,
    sample_rate: int,
    segment_duration: float,
    frame: int,
    hop: int,
    n_fft: int,
    high_pass_filter: int = 0,
    dc_correction: int = 0,
    min_conf: float = 0.0,
    partition_size: int = None,
    npartitions: int = None,
    **kwargs: Any,
) -> Tuple[dd.DataFrame, dd.Scalar] | pd.DataFrame:
    # ensure dataset class for parsing relevant information has been setup
    assert dataset in datasets, f"Unsupported dataset '{dataset}'"
    # setup data sinks
    save_dir = save_dir.expanduser()
    save_dir.mkdir(exist_ok=True, parents=True)
    sites_path = sitesfile or save_dir / "locations_table.parquet"
    files_path = save_dir / "files_table.parquet"
    solar_path = save_dir / "solar_table.parquet"
    recording_acoustic_features_path = save_dir / "recording_acoustic_features_table.parquet"
    birdnet_species_probs_path = save_dir / "birdnet_species_probs_table.parquet"
    # begin timing
    start_time = time.time()
    # index sites
    # NB: this just resaves the parquet file and is effectively redundant
    # however is left there incase custom behaviour by dataset is required
    log.info(f"Processing site information")
    sites_df = index_sites(
        root_dir=root_dir,
        out_file=sites_path,
        dataset=dataset,
    )
    # index files
    log.info(f"Indexing audio files")
    files_df, _ = index_audio(
        root_dir=root_dir,
        out_file=files_path,
        sites_ddf=dd.from_pandas(sites_df),
        dataset=dataset,
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
    log.info(f"Indexing weather data")
    index_weather(
        files_df=files_df,
        sites_df=sites_df,
        save_dir=save_dir,
    )
    # extract acoustic featres
    log.info(f"Extracting acoustic features")
    acoustic_features_ddf, acoustic_features_future = acoustic_features(
        root_dir=root_dir,
        files_df=files_df,
        outfile=recording_acoustic_features_path,
        sample_rate=sample_rate,
        frame=frame,
        hop=hop,
        n_fft=n_fft,
        segment_duration=segment_duration,
        high_pass_filter=high_pass_filter,
        dc_correction=dc_correction,
        compute=False,
    )
    # extract birdnet species scores
    log.info(f"Extracting BirdNET species probabilities")
    birdnet_species_ddf, birdnet_species_future = birdnet_detections(
        root_dir=root_dir,
        files_df=files_df,
        sites_df=sites_df,
        outfile=birdnet_species_probs_path,
        min_conf=min_conf,
        compute=False,
    )
    # compute the graph
    log.info(f"Processing...")
    dask.compute(acoustic_features_future, birdnet_species_future)
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
        '--dataset',
        type=str,
        choices=datasets.keys(),
        help='Name of the dataset',
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        help='Resample rate for audio',
    )
    parser.add_argument(
        '--segment-duration',
        type=float,
        default=60.0,
        help='Duration for chunking audio segments. Defaults to 60s. Specify -1 to use full clip.',
    )
    parser.add_argument(
        '--frame',
        type=int,
        help='Number of audio frames for a feature frame.',
    )
    parser.add_argument(
        '--hop',
        type=int,
        help='Number of audio frames for the hop.',
    )
    parser.add_argument(
        '--n-fft',
        type=int,
        help='Number of audio frames for the n_fft.',
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
        "--min-conf",
        type=float,
        help="BirdNET confidence threshold. Defaults to 0.0 to collect all detections.",
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
    parser.set_defaults(func=main, **{
        "root_dir": "/data",
        "save_dir": "/results",
        "dataset": os.environ.get("DATASET", None),

        'local': os.environ.get("LOCAL", True),
        "memory": os.environ.get("MEM_PER_CPU", 0),
        "cores": os.environ.get("CORES", 0),
        "threads_per_worker": os.environ.get("THREADS_PER_WORKER", 1),

        "sample_rate": os.environ.get("SAMPLE_RATE", 48_000),
        "segment_duration": os.environ.get("SEGMENT_LEN", 60.0),
        "frame": os.environ.get("FRAME", 2_048),
        "hop": os.environ.get("HOP", 512),
        'n_fft': os.environ.get("N_FFT", 2_048),
        "dc_correction": os.environ.get("DC_CORR", 0),
        "high_pass_filter": os.environ.get("HIGH_PASS_FILTER", 0),

        "min_conf": os.environ.get("MIN_CONF", 0.0),
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
