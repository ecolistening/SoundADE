import pandas as pd
from dask import dataframe as dd
from dask.distributed import Client, LocalCluster
import pyarrow as pa

from soundade.data.filter import channels, first_n_days, days_with_too_few_points
from soundade.hpc.cluster import AltairGridEngineCluster
from soundade.hpc.arguments import DaskArgumentParser

from pathlib import Path

def main(infile=None, outfile=None, n_days=10, memory=64, cores=4, jobs=1, npartitions=None,
    local=True, test=False, **kwargs):
    """
  Main function for prefiltering data to create a set of data with the same number of days across sites.
  This script is specific to the DAC project data and is not intended for general use.

  Args:
    infile (str): Path to the input file.
    outfile (str): Path to the output file.
    memory (int): Amount of memory to allocate for the cluster in GB.
    cores (int): Number of CPU cores to use for the cluster.
    jobs (int): Number of jobs to run on the cluster.
    npartitions (int): Number of partitions to use for the input data.
    local (bool): Flag indicating whether to run the cluster locally.
    test (bool): Flag indicating whether to run in test mode.
    **kwargs: Additional keyword arguments.

  Returns:
    None

  Examples:
    >>> main(infile='~/data/processed/ecolistening/features.solar.parquet',
    ...  outfile='~/data/processed/ecolistening/features.prefilter.parquet',
    ...  n_days=None, local=True)
  """
  
    assert infile is not None
    assert outfile is not None
    
    outfile = Path(outfile)

    if not local:
        # Start cluster
        cluster = AltairGridEngineCluster(cores=cores, memory=memory, queue='test.short', name=None)#.short', name=None)
        print(cluster.job_script())
        cluster.scale(jobs=jobs)
        client = Client(cluster)
    else:
        memory_per_worker = f'{memory}GiB'
        client = Client(n_workers=cores,
                        threads_per_worker=1,
                        memory_limit=memory_per_worker)
        print(client)

    # Read data
    ddf = dd.read_parquet(infile)
    if npartitions is not None:
        ddf = ddf.repartition(npartitions=npartitions)
    ddf = ddf.persist()

    ddf['date'] = dd.to_datetime(ddf.timestamp.dt.date)

    # Select Channels
    ddf = ddf.map_partitions(channels)

    # Remove first and last days
    # ddf = first_and_last_days(ddf, dask=True)
    # Remove days that don't have the same number of data points as the rest of the days from that location
    ddf = days_with_too_few_points(ddf, groupby=['location'], agg_columns=['date'], dask=True)
    dd.to_parquet(ddf.drop(columns='date'), outfile.with_stem(f'{outfile.stem}_prefilter_1_days'), version='2.6',
                  allow_truncated_timestamps=True)

    # Select first 10 days
    if n_days is not None:
        ddf = first_n_days(ddf, n=n_days, dask=True).persist()
        print('First n Days (head): ', ddf.head())
        dd.to_parquet(ddf.drop(columns='date'), outfile.with_stem(f'{outfile.stem}_prefilter_2_first_days'), version='2.6',
                allow_truncated_timestamps=True)

    # Remove recorders with missing data points
    ddf = days_with_too_few_points(ddf, groupby=['location'], agg_columns=['recorder'], dask=True) # recorders_with_too_few_points(ddf)

    ddf = ddf.drop(columns=['date', '0', '1', '2', '717', '718', '719']).persist().repartition(partition_size="20MB")

    dd.to_parquet(ddf, outfile, version='2.6', allow_truncated_timestamps=True)#, schema={'date': pa.date32(), 'time': pa.time64('ns')})


if __name__ == '__main__':
    parser = DaskArgumentParser('Extract features from audio files', memory=128, cores=1, jobs=4, npartitions=None)
    
    parser.add_argument('--n-days', default=10, help='How many days to select from the data.')
    
    args = parser.parse_args()
    print(args)

    main(**vars(args))
