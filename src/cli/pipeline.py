import os
import argparse
import dask
import logging
import pandas as pd
import time

from dask import bag as db
from dask import dataframe as dd
from dask import config as cfg
from dask.distributed import Client
from pathlib import Path
from typing import Any, Tuple

from soundade.hpc.arguments import DaskArgumentParser
from soundade.data.bag import file_path_to_audio_dict
from soundade.datasets import datasets

from cli.index_sites import index_sites
from cli.index_audio import index_audio
from cli.index_solar import index_solar
from cli.acoustic_features import acoustic_features
from cli.birdnet_species import birdnet_species

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

cfg.set({
    "distributed.scheduler.worker-ttl": None
})

def pipeline(
    root_dir: str | Path,
    save_dir: str | Path,
    sitesfile: str | Path | None,
    dataset: str,
    segment_duration: float = 60.0,
    frame: int = 0,
    hop: int = 0,
    n_fft: int = 0,
    min_conf: float = 0.3,
    partition_size: int = None,
    npartitions: int = None,
) -> Tuple[dd.DataFrame, dd.Scalar] | pd.DataFrame:
    # ensure dataset class for parsing relevant information has been setup
    assert dataset in datasets, f"Unsupported dataset '{dataset}'"
    # setup data sinks
    save_dir.mkdir(exist_ok=True, parents=True)
    sites_path = sitesfile or save_dir / "locations.parquet"
    files_path = save_dir / "files.parquet"
    solar_parth = save_dir / "solar.parquet"
    recording_acoustic_features_path = save_dir / "recording_acoustic_features.parquet"
    birdnet_species_probs_path = save_dir / "birdnet_species_probs.parquet"
    # begin timing
    start_time = time.time()
    # index sites if not already available
    log.info(f"Processing site information")
    sites_df = index_sites(
        root_dir=root_dir,
        out_file=sites_path,
        dataset=dataset,
    ) if not sitesfile else pd.read_parquet(sitesfile)
    # index files
    log.info(f"Indexing audio files")
    files_df = index_audio(
        root_dir=root_dir,
        out_file=files_path,
        sites=sites_df,
        dataset=dataset,
        compute=True,
    )
    # index solar times
    log.info(f"Indexing solar times")
    files_df, solar_df = index_solar(
        files=dask.dataframe.from_pandas(files_df),
        sites=sites_df,
        infile=save_dir / "files.parquet",
        outfile=save_dir / "solar.parquet",
        compute=True,
    )
    # extract acoustic featres
    log.info(f"Extracting acoustic features")
    acoustic_features_ddf, acoustic_features_future = acoustic_features(
        files=dask.dataframe.from_pandas(files_df),
        outfile=recording_acoustic_features_path,
        segment_duration=segment_duration,
        frame=frame,
        hop=hop,
        n_fft=n_fft,
        compute=False,
    )
    # extract birdnet species scores
    log.info(f"Extracting BirdNET species probabilities")
    birdnet_species_ddf, birdnet_species_future = birdnet_species(
        files=dask.dataframe.from_pandas(files_df),
        sites=sites_df,
        outfile=birdnet_species_probs_path,
        min_conf=min_conf,
        compute=False,
    )
    # compute the graph
    log.info(f"Processing...")
    dask.compute(
        acoustic_features_future,
        birdnet_species_future,
    )
    # and we're done!
    log.info(f"Pipeline complete, time taken: {time.time() - start_time}")

def main(
    root_dir: str | Path,
    save_dir: str | Path,
    sitesfile: str | Path | None = None,
    dataset: str = None,
    segment_duration: float = 60.0,
    frame: int = 16_000,
    hop: int = 4_000,
    n_fft: int = 1024,
    min_conf: float = 0.3,
    memory: int = 4,
    cores: int = 1,
    jobs: int = 0,
    queue: str = 'general',
    npartitions: int | None = None,
    local: bool = True,
    threads_per_worker: int = 1,
    debug: bool = False,
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

    pipeline(
        root_dir=root_dir,
        save_dir=save_dir,
        sitesfile=sitesfile,
        dataset=dataset,
        segment_duration=segment_duration,
        frame=frame,
        hop=hop,
        n_fft=n_fft,
        min_conf=min_conf,
        npartitions=npartitions,
    )

def get_base_parser():
    parser = DaskArgumentParser(
        description="Run the full pipeline",
        add_help=False,
    )
    parser.add_argument(
        "--root-dir",
        type=lambda p: Path(p),
        default=os.environ.get("DATA_PATH", "/data"),
        help="Root directory containing audio files",
    )
    parser.add_argument(
        '--save-dir',
        type=lambda p: Path(p),
        default=os.environ.get("DATA_PATH", "/data"),
        help='Target directory for results',
    )
    parser.add_argument(
        '--sitesfile',
        type=lambda p: Path(p),
        default="/".join([os.environ.get("DATA_PATH", "/data"), "locations_table.parquet"]),
        default=None,
        help='Refencing a locations.parquet with site-level info (site_name/lat/lng/etc)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=os.environ.get("DATASET"),
        choices=datasets.keys(),
        help='Name of the dataset',
    )
    parser.add_argument(
        '--segment-duration',
        default=60.0,
        type=float,
        help='Duration for chunking audio segments (defaults to 60s). Specify -1 to use full clip.'
    )
    parser.add_argument(
        '--frame',
        type=int,
        help='Number of audio frames for a feature frame.'
    )
    parser.add_argument(
        '--hop',
        type=int,
        help='Number of audio frames for the hop.'
    )
    parser.add_argument(
        '--n-fft',
        type=int,
        help='Number of audio frames for the n_fft.'
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.5,
        help="BirdNET confidence threshold"
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="Threads per worker",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Sets single-threaded for debugging.",
    )
    parser.set_defaults(func=main, **{
        "root_dir": os.environ.get("DATA_PATH", "/data"),
        "memory": os.environ.get("MEM_PER_CPU", 0),
        "cores": os.environ.get("CORES", 1),
        "threads_per_worker": 1,
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
