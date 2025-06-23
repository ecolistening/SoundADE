import argparse
import time
import logging
import pandas as pd

from dask import bag as db
from dask import dataframe as dd
from dask import config as cfg
from dask.distributed import Client
from pathlib import Path
from typing import Any, Tuple

from soundade.hpc.arguments import DaskArgumentParser
from soundade.data.bag import file_path_to_audio_dict
from soundade.datasets import datasets

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

cfg.set({
    "distributed.scheduler.worker-ttl": None
})

def file_meta():
    return pd.DataFrame({
        "file_cid": pd.Series(dtype="string[pyarrow]"),
        "file_name": pd.Series(dtype="string[pyarrow]"),
        "local_file_path": pd.Series(dtype="string[pyarrow]"),
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
    sites: pd.DataFrame | dd.DataFrame,
    dataset: str,
    partition_size: int = None,
    npartitions: int = None,
    compute: bool = True,
) -> Tuple[dd.DataFrame, dd.Scalar] | pd.DataFrame:
    assert dataset in datasets, f"Unsupported dataset '{dataset}'"
    dataset: Dataset = datasets[dataset]

    start_time = time.time()

    root_dir = Path(root_dir)

    log.info("Recursively discovering audio files...")
    file_list = list(root_dir.rglob("*.[wW][aA][vV]"))

    # TODO: move to file indexing stage
    # log.info(f"Applying dataset specific filter...")
    # b = b.filter(dataset.prefilter_file_dictionary)

    log.info(f"{len(file_list)} audio files found. Building file index...")
    ddf = (
        db.from_sequence(file_list, partition_size=partition_size, npartitions=npartitions)
        # extract audio metadata and attach 'valid' indicator of corrupt audio
        .map(file_path_to_audio_dict)
        # attach site name from audio path, e.g. Knepp/S_SW1, Cairngorms/Wood, sounding_out/1/1/15
        .map(dataset.extract_site_name)
        # attach timestamp from end of audio path
        .map(dataset.extract_timestamp)
        # convert to dataframe
        .to_dataframe(meta=file_meta())
    )

    # attach site_id as key reference and drop site name
    if sites is not None:
        ddf = (
            ddf
            .merge(sites[["site_name", "site_id"]], on="site_name", how="left")
            .drop("site_name", axis=1)
        )

    # compute immediately
    if compute:
        df = ddf.compute()
        df.to_parquet(Path(out_file), index=False)
        log.info(f"Time taken for file index: {time.time() - start_time}")
        log.info(f"File index saved to {out_file}")
        return df

    future = ddf.to_parquet(
        Path(out_file),
        version='2.6',
        allow_truncated_timestamps=True,
        write_index=False,
        compute=False,
    )

    log.info(f"Indexing files queued, will persist to {out_file}")

    return ddf, future

def main(
    root_dir: str | Path,
    out_file: str | Path,
    sitesfile: str | Path | None,
    dataset: str = None,
    memory: int = 4,
    cores: int = 1,
    threads_per_worker: int = 1,
    compute: bool = False,
    debug: bool = False,
    **kwargs: Any,
) -> None:
    """
    Build files table, indexing, validating audio files in the specified directory and attaching site reference.

    Args:
        root_dir (str, required): Input directory containing audio files.
        outfile (str, required): Output file path.
        sitesfile (str, optional): Sites file stores a foreign key for a sites table,
                                   where the join key is extracted from the dataset's path.
                                   Required for secondary pipeline steps: birdnet and solar.
        memory (int, optional): Memory limit for each worker in GB. Defaults to 32.
        cores (int, optional): Number of CPU cores per worker. Defaults to 8.
        compute (bool, optional): Compute and save the parquet. Defaults to true. Returns a Dask Dataframe if false.
        debug (bool, optional): Flag to set processing to synchronous for debugging. Defaults to False.

    Returns:
        dd.DataFrame
    """
    if debug:
        cfg.set(scheduler='synchronous')

    client = Client(
        n_workers=cores,
        threads_per_worker=threads_per_worker,
        memory_limit=f'{memory}GiB',
    )
    log.info(client)

    index_audio(
        root_dir=root_dir,
        out_file=out_file,
        dataset=dataset,
        sites=dd.read_parquet(sitesfile),
    )

def get_base_parser():
    parser = argparse.ArgumentParser(
        description="Index audio files",
        add_help=False,
    )
    parser.add_argument(
        "--root-dir",
        type=lambda p: Path(p),
        required=True,
        help="Root directory containing (1) a locations.parquet file and (2) audio files (nested folder structure permitted)",
    )
    parser.add_argument(
        '--out-file',
        type=lambda p: Path(p),
        required=True,
        help='Parquet file to save results.',
    )
    parser.add_argument(
        '--sitesfile',
        type=lambda p: Path(p),
        required=False,
        help='Refencing a locations.parquet with site-level info (site_name/lat/lng/etc)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=datasets.keys(),
        help='Name of the dataset',
    )
    parser.add_argument(
        "--memory",
        default=8,
        type=int,
        help="Amount of memory required in GB (total per node).",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of cores per node.",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
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
        "root_dir": "/data",
        "out_file": "/data/files_table.parquet",
        "memory": 0,
        "cores": 1,
        "threads": 1,
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
