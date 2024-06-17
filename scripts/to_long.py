from pathlib import Path

import dask.dataframe as dd
from dask.distributed import Client
import numpy as np

from soundade.hpc.cluster import AltairGridEngineCluster
from soundade.hpc.arguments import DaskArgumentParser

import logging
logging.basicConfig(level=logging.INFO)

def main(infile=None, outfile=None, memory=24, cores=4, jobs=8, npartitions=None,
     test=False, queue='test.short', binary=True, local=False, first_frame='0', **kwargs):
  """
  Convert a wide-form dataframe to long-form and save it as a parquet file.

  Parameters:
  infile (str): Path to the input parquet file.
  outfile (str): Path to save the output parquet file.
  memory (int): Amount of memory (in GB) to allocate for the cluster.
  cores (int): Number of CPU cores to use for the cluster.
  jobs (int): Number of jobs to run in parallel on the cluster.
  npartitions (int): Number of partitions to create for the dataframe.
  test (bool): Flag indicating whether to run in test mode.
  queue (str): Name of the queue to submit jobs to.
  binary (bool): Flag indicating whether to use binary format for parquet files.
  local (bool): Flag indicating whether to run in local mode.
  first_frame (str): The first frame to include in the long-form dataframe.
  **kwargs: Additional keyword arguments.

  Returns:
  None

  Examples:
      >>> main(infile='./data/processed/ecolistening/features.solar.parquet',
      ...      outfile='./data/processed/ecolistening/features.long.parquet',
      ...      dataset='SoundingOutDiurnal', first_frame='3',
      ...      local=True)

  TODO: first_frame is a bit of a hack. It drops the first few frames of data which are often skewed, depending on the feature. However, this should be done elsewhere and should be a number not a string.
  """
  
  assert infile is not None
  assert outfile is not None

  outfile = Path(outfile)

  if not local:
    # Start cluster
    cluster = AltairGridEngineCluster(cores=cores, memory=memory, queue='test.short',
                      name=None)  # .short', name=None)
    logging.info(cluster.job_script())
    cluster.scale(jobs=jobs)
    client = Client(cluster)
  else:
    client = Client()
    logging.info(client)

  # Read data
  df = dd.read_parquet(infile)

  logging.info('Initial Load')

  # Melt to long form
  metadata = df.iloc[:, :df.columns.get_loc(str(first_frame))]
  features = df.iloc[:, df.columns.get_loc(str(first_frame)):]

  df_long = df.melt(id_vars=metadata.columns, value_vars=features.columns,
            var_name='frame', value_name='value')
  df_long = df_long.assign(idx=df_long.feature.astype(str) + ' ' + df_long.recorder.astype(str) + ' ' + df_long.timestamp.dt.date.astype(str))
  df_long = df_long.dropna(subset='value')
  df_long = df_long.set_index('idx')

  # Date columns might not work for this...
  logging.info(f'Outputting ddf to parquet')
  dd.to_parquet(df_long, outfile, version='2.6', allow_truncated_timestamps=True)

if __name__ == '__main__':
    parser = DaskArgumentParser('Compute astrograms from audio feature sets', memory=48, cores=1, jobs=95,
                                npartitions=None, queue='test.short')
    
    parser.add_argument('--first-frame', default='0', type=str,
                          help='First frame')

    args = parser.parse_args()
    print(args)

    main(**vars(args))
