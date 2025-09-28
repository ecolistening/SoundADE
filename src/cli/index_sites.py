import argparse
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
from soundade.data.bag import file_path_to_audio_dict
from soundade.data.dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def index_sites(
    root_dir: str | Path,
    config_path: Path,
    out_file: str | Path,
    **kwargs: Any,
) -> pd.DataFrame:
    dataset = Dataset.from_config_path(config_path)

    df = dataset.index_sites(root_dir, out_file)

    df.to_parquet(Path(out_file))
    log.info(f"Sites index saved to {out_file}")

    return df

def main(**kwargs: Any) -> None:
    """
    Build sites table extract location information from the audio directory

    Args:
        root_dir (str, required): Input directory containing site metadata
        outfile (str, required): Output file path.
        dataset (str, required): Dataset to use

    Returns:
        pd.DataFrame
    """
    start_time = time.time()

    index_sites(**kwargs)

    log.info(f"Site index complete")
    log.info(f"Time taken: {str(dt.timedelta(seconds=time.time() - start_time))}")

def get_base_parser():
    parser = argparse.ArgumentParser(
        description="Index site information",
        add_help=False,
    )
    parser.add_argument(
        "--root-dir",
        type=lambda p: Path(p).expanduser(),
        help="Root directory containing (1) a locations.parquet file and (2) audio files (nested folder structure permitted)",
    )
    parser.add_argument(
        '--config-path',
        type=lambda p: Path(p).expanduser(),
        help='/path/to/dataset/config.yaml',
    )
    parser.add_argument(
        '--out-file',
        type=lambda p: Path(p).expanduser(),
        help='Parquet file to save results.',
    )
    parser.set_defaults(func=main, **{
        "root_dir": "/data",
        "config_path": "/config.yml",
        "out_file": "/results/locations_table.parquet",
    })
    return parser

def register_subparser(subparsers):
    parser = subparsers.add_parser(
        "index_sites",
        help="Index site information",
        parents=[get_base_parser()],
        add_help=True,
    )

if __name__ == "__main__":
    parser = get_base_parser()
    args = parser.parse_args()
    log.info(args)
    args.func(**vars(args))
