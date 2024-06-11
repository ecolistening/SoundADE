import pandas as pd
from dask import dataframe as dd
from dask.distributed import Client, LocalCluster
import pyarrow as pa

from soundade.data.datasets import SoundingOutDiurnal
from soundade.hpc.cluster import AltairGridEngineCluster
from soundade.data.solar import solartimes
from soundade.hpc.arguments import DaskArgumentParser

#TODO test parameter does nothing. Remove.

def main_local(infile=None, outfile=None, sitesfile=None, memory=64, cores=4, jobs=2, npartitions=20,
         filename=True, timeparts=True, country_habitat=True, solar=True, compute=False,
         test=False, **kwargs):
    assert infile is not None
    assert outfile is not None

    # Read data
    df = pd.read_parquet(infile)

    cols_meta = list(df.columns[:3])
    cols_data = list(df.columns[3:])

    if filename:
        df = SoundingOutDiurnal.filename_metadata(df, cols_data)

    if timeparts:
        df = SoundingOutDiurnal.timeparts(df)

    if country_habitat:
        df = SoundingOutDiurnal.country_habitat(df, use_meta=False)

    if solar:
        df = solartimes(df, locations=sitesfile)

    df.to_parquet(outfile)


def main(infile=None, outfile=None, sitesfile=None, memory=64, cores=4, jobs=2, npartitions=20,
         filename=True, timeparts=True, country_habitat=True, solar=True, compute=False,
         test=False, local_cluster=False, **kwargs):
    """
    Process and append metadata to the input data.

    Args:
        infile (str): Path to the input data file.
        outfile (str): Path to the output file where the processed data will be saved.
        memory (int): Amount of memory (in GB) to allocate for the computation.
        cores (int): Number of CPU cores to use for the computation.
        jobs (int): Number of parallel jobs to run.
        npartitions (int): Number of partitions to repartition the data into.
        filename (bool): Flag indicating whether to append filename metadata to the data.
        timeparts (bool): Flag indicating whether to process time parts of the data.
        country_habitat (bool): Flag indicating whether to process country and habitat information of the data.
        solar (bool): Flag indicating whether to process solar information of the data.
        compute (bool): Flag indicating whether to compute and save the processed data.
        test (bool): Flag indicating whether to run the function in test mode.
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
        cluster = LocalCluster()
        client = Client(cluster)
    else:
        # Start cluster
        cluster = AltairGridEngineCluster(cores=cores, memory=memory, queue='test.short', name=None)  # 'AppendMetadata')
        print(cluster.job_script())
        client = Client(cluster)
        cluster.scale(jobs=jobs)

    # Read data
    df = dd.read_parquet(infile)
    print(f'Before repartition: {df.npartitions}')
    # if npartitions is not None:
    #     print(f'Repartitioning to {npartitions} partitions')
    #     df = df.repartition(npartitions=npartitions)
    df = df.persist()
    df = df.repartition(partition_size='20MB').persist()
    print(f'After repartition: {df.npartitions}')

    cols_meta = list(df.columns[:3])
    cols_data = list(df.columns[3:])
    
    if filename:
        df = SoundingOutDiurnal.filename_metadata(df, cols_data)

    # TODO DawnDusk flag
    
    #TODO this might be causing problems.
    if timeparts:
        df = SoundingOutDiurnal.timeparts(df)

    if country_habitat:
        df = SoundingOutDiurnal.country_habitat(df)
    
    if solar:
        df = SoundingOutDiurnal.solar(df, locations=sitesfile)
    
    df = df.persist()
    
    if compute:
        df.compute().to_parquet(outfile)
    else:
        dd.to_parquet(df, outfile, version='2.6', write_index=False, allow_truncated_timestamps=True)#, schema={'date': pa.date32(), 'time': pa.time64('ns')})
        # dd.to_parquet(df, outfile, write_index=False)

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

    parser.add_argument('--local-cluster', default=False, action='store_true')

    args = parser.parse_args()
    print(args)

    if args.local:
        main_local(**vars(args))
    else:
        main(**vars(args))
