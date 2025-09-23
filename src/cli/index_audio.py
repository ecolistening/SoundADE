import argparse
import dask
import datetime as dt
import itertools
import logging
import os
import pandas as pd
import time

from dask import bag as db
from dask import dataframe as dd
from dask.distributed import Client
from pathlib import Path
from typing import Any, Tuple

from soundade.data.bag import file_path_to_audio_dict
from soundade.datasets import datasets

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

PYARROW_VERSION = "2.6"

def file_meta():
    return pd.DataFrame({
        "file_id": pd.Series(dtype="string[pyarrow]"),
        "file_name": pd.Series(dtype="string[pyarrow]"),
        "file_path": pd.Series(dtype="string[pyarrow]"),
        "size": pd.Series(dtype="int64[pyarrow]"),
        "valid": pd.Series(dtype="bool[pyarrow]"),
        "duration": pd.Series(dtype="float64[pyarrow]"),
        "sr": pd.Series(dtype="int32[pyarrow]"),
        "channels": pd.Series(dtype="int8[pyarrow]"),
        "site_name": pd.Series(dtype="string[pyarrow]"),
        "timestamp": pd.Series(dtype="datetime64[us]"),
    })

def index_audio(
    root_dir: str | Path,
    out_file: str | Path,
    sites_ddf: pd.DataFrame | dd.DataFrame | None,
    dataset: str,
    partition_size: int = None,
    npartitions: int = None,
    compute: bool = True,
) -> Tuple[dd.DataFrame, dd.Scalar] | pd.DataFrame:
    assert dataset in datasets, f"Unsupported dataset '{dataset}'"

    if sites_ddf is not None:
        assert "site_id" in sites_ddf.columns, f"'site_id' key must be available in the sites table"
        assert "site_name" in sites_ddf.columns, f"'site_name' must be available in the sites table and should align with site directory structure"

    root_dir = Path(root_dir).expanduser()
    dataset: Dataset = datasets[dataset]()

    log.info("Recursively discovering audio files...")

    wavs = root_dir.rglob("*.[wW][aA][vV]")
    mp3s = root_dir.rglob("*.[mM][pP]3")
    flacs = root_dir.rglob("*.[fF][lL][aA][cC]")
    file_paths = list(itertools.chain(wavs, mp3s, flacs))
    assert len(file_paths), f"No audio files found at {root_dir}"
    file_paths = [file_path.relative_to(root_dir) for file_path in file_paths]
    b = db.from_sequence(file_paths, partition_size=partition_size, npartitions=npartitions)

    log.info(f"{len(file_paths)} audio files found. Building file index...")
    log.info(f"Partitions: {b.npartitions}")

    files_ddf = (
        # extract audio metadata and attach 'valid' indicator of corrupt audio
        b.map(file_path_to_audio_dict, root_dir=root_dir)
        # attach site hierarchy from audio path
        # e.g. kilpisjarvi/K1, nature_sense/Knepp/S_SW1, cairngorms/Wood, sounding_out/uk/1/15
        .map(dataset.extract_site_name)
        # attach timestamp from end of audio path
        .map(dataset.extract_timestamp)
        # convert to dataframe
        .to_dataframe(meta=file_meta())
    )

    # attach site_id as foreign key
    if sites_ddf is not None:
        files_ddf = (
            files_ddf
            .merge(sites_ddf[["site_name", "site_id"]], on="site_name", how="left")
            .drop("site_name", axis=1)
            # opinion: drop files that haven't been assigned a site
            .dropna(subset="site_id")
            .astype({"site_id": "string[pyarrow]"})
        )

    future = files_ddf.to_parquet(
        Path(out_file),
        version=PYARROW_VERSION,
        allow_truncated_timestamps=True,
        write_index=False,
        compute=False,
    )

    log.info(f"Indexing files queued, will persist to {out_file}")

    # compute immediately
    if compute:
        files_df = dask.compute(future)
        log.info(f"File index saved to {out_file}")
        return pd.read_parquet(Path(out_file)), None

    return ddf, future

def main(
    root_dir: str | Path,
    out_file: str | Path,
    sitesfile: str | Path | None,
    dataset: str = None,
    memory: int = 4,
    cores: int = 1,
    threads_per_worker: int = 1,
    compute: bool = True,
    debug: bool = False,
    **kwargs: Any,
) -> None:
    """
    Build files table, indexing, validating audio files in the specified directory and attaching site reference.

    Args:
        root_dir (str, required): Input directory containing audio files.
        out-file (str, required): Output file path.
        sitesfile (str, optional): Sites file stores a foreign key for a sites table,
                                   where the join key is extracted from the dataset's path.
                                   Required for secondary pipeline steps: birdnet and solar.
        memory (int, optional): Memory limit for each worker in GB. Defaults to 32.
        cores (int, optional): Number of CPU cores per worker. Defaults to 8.
        compute (bool, optional): Compute and save the parquet. Defaults to true. Returns a Dask Dataframe if false.
        debug (bool, optional): Flag to set processing to synchronous for debugging. Defaults to False.

    Returns:
        None
    """
    if debug:
        cfg.set(scheduler='synchronous')

    client = Client(
        n_workers=cores,
        threads_per_worker=threads_per_worker,
        memory_limit=f'{memory}GiB',
    )
    log.info(client)

    start_time = time.time()

    index_audio(
        root_dir=root_dir,
        out_file=out_file,
        dataset=dataset,
        sites_ddf=dd.read_parquet(sitesfile),
    )

    log.info(f"File index complete")
    log.info(f"Time taken: {str(dt.timedelta(seconds=time.time() - start_time))}")

def get_base_parser():
    parser = argparse.ArgumentParser(
        description="Index audio files",
        add_help=False,
    )
    parser.add_argument(
        "--root-dir",
        type=lambda p: Path(p).expanduser(),
        help="Root directory of the audio files (nested folder structure permitted)",
    )
    parser.add_argument(
        '--out-file',
        type=lambda p: Path(p).expanduser(),
        help='Parquet file to save results.',
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
        "--memory",
        type=int,
        help="Amount of memory required in GB (total per node).",
    )
    parser.add_argument(
        "--cores",
        type=int,
        help="Number of cores per node.",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        help="Threads per worker",
    )
    parser.add_argument(
        "--compute",
        default=True,
        action="store_true",
        help="Compute and save to parquet.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Sets single-threaded for debugging.",
    )
    parser.set_defaults(func=main, **{
        "root_dir": os.environ.get("DATA_PATH", "/data"),
        "out_file": "/".join([os.environ.get("DATA_PATH", "/data"), "files_table.parquet"]),
        "sitesfile": "/".join([os.environ.get("DATA_PATH", "/data"), "locations_table.parquet"]),
        "dataset": os.environ.get("DATASET", None),
        "memory": os.environ.get("MEM_PER_CPU", 0),
        "cores": os.environ.get("CORES", 1),
        "threads_per_worker": 1,
    })
    return parser

def register_subparser(subparsers):
    parser = subparsers.add_parser(
        "index_audio",
        help="Index audio files",
        parents=[get_base_parser()],
        add_help=True,
    )

if __name__ == "__main__":
    parser = get_base_parser()
    args = parser.parse_args()
    log.info(args)
    args.func(**vars(args))
