import datetime as dt
import numpy as np
import pandas as pd
import pyarrow as pa
import logging

from dask import dataframe as dd
from dask.distributed import Client, LocalCluster
from pathlib import Path

from soundade.datasets import datasets
from soundade.datasets.base import Dataset
from soundade.datasets.sounding_out_diurnal import SoundingOutDiurnal
from soundade.hpc.cluster import AltairGridEngineCluster
from soundade.data.solar import solartimes
from soundade.hpc.arguments import DaskArgumentParser

logging.basicConfig(level=logging.INFO)

def main(
    infile=None,
    outfile=None,
    sitesfile=None,
    memory=64,
    cores=4,
    jobs=1,
    npartitions=None,
    dataset=None,
    filename=True,
    timeparts=True,
    country_habitat=True,
    solar=True,
    compute=False,
    local_cluster=True,
    **kwargs
):
    """
    Process and append metadata to the input data.

    Args:
        infile (str): Path to the input data file.
        outfile (str): Path to the output file where the processed data will be saved.
        memory (int): Amount of memory (in GB) to allocate for the computation.
        cores (int): Number of CPU cores to use for the computation.
        jobs (int): Number of parallel jobs to run.
        npartitions (int): Number of partitions to repartition the data into.
        dataset (str, optional): Name of the dataset to use. The name of a dataset class defined in soundade.datasets. Required.
        filename (bool): Flag indicating whether to append filename metadata to the data.
        timeparts (bool): Flag indicating whether to process time parts of the data.
        country_habitat (bool): Flag indicating whether to process country and habitat information of the data.
        solar (bool): Flag indicating whether to process solar information of the data.
        compute (bool): Flag indicating whether to compute and save the processed data. Used mainly for very small datasets and for testing purposes.
        local_cluster (bool): Flag indicating whether to use a local cluster for computation.

    Returns:
        None

    Examples:
        >>> main(infile='./data/processed/ecolistening/features.parquet', outfile='./data/processed/ecolistening/features.solar.parquet',
        ...      sitesfile='./data/ecolistening/site_locations.parquet', filename=False, timeparts=False, country_habitat=False, local_cluster=True, compute=True)
        Before repartition: ...
    """

    assert infile is not None
    assert outfile is not None

    if local_cluster:
        memory_per_worker = f'{memory}GiB'
        client = Client(
            n_workers=cores,
            threads_per_worker=1,
            memory_limit=memory_per_worker
        )
    else:
        cluster = AltairGridEngineCluster(
            cores=cores,
            memory=memory,
            queue='test.short',
            name=None
        )
        print(cluster.job_script())
        client = Client(cluster)
        cluster.scale(jobs=jobs)

    assert dataset in datasets, f"Unsupported dataset '{dataset}'"
    ds: Dataset = datasets[dataset]

    outfile = Path(outfile)
    ddf = dd.read_parquet(infile)
    ddf = ds.metadata(ddf)
    import code; code.interact(local=locals())
    # ddf = ds.time_parts(ddf)
    # ddf = ds.solar(ddf, locations=sitesfile)

    if compute:
        df = ddf.compute()
        df.to_parquet(outfile)
    else:
        dd.to_parquet(
            ddf,
            outfile,
            version='2.6',
            write_index=False,
            allow_truncated_timestamps=True,
        )

if __name__ == '__main__':
    parser = DaskArgumentParser('Extract features from audio files', memory=128, cores=1, jobs=4, npartitions=None)

    parser.add_argument('--sitesfile', default=None, help='Parquet file containing site information.')

    filename = parser.add_mutually_exclusive_group()
    filename.add_argument('-f', dest='filename', default=True, action='store_true', help='Process filename metadata')
    filename.add_argument('-F', dest='filename', default=True, action='store_false', help='Do not process filename metadata')

    country_habitat = parser.add_mutually_exclusive_group()
    country_habitat.add_argument('-c', dest='country_habitat', default=True, action='store_true')
    country_habitat.add_argument('-C', dest='country_habitat', default=True, action='store_false')

    solar = parser.add_mutually_exclusive_group()
    solar.add_argument('-s', dest='solar', default=True, action='store_true')
    solar.add_argument('-S', dest='solar', default=True, action='store_false')

    timeparts = parser.add_mutually_exclusive_group()
    timeparts.add_argument('-t', dest='timeparts', default=True, action='store_true')
    timeparts.add_argument('-T', dest='timeparts', default=True, action='store_false')

    parser.add_argument('--compute', dest='compute', default=False, action='store_true')

    parser.add_argument('--local-cluster', default=True, action='store_true')

    args = parser.parse_args()
    print(args)

    main(**vars(args))
