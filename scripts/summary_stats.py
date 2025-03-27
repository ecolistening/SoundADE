import logging
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import scipy.stats
from dask.distributed import Client

from soundade.hpc.cluster import AltairGridEngineCluster
from soundade.hpc.arguments import DaskArgumentParser

logging.basicConfig(level=logging.INFO)

def sstats(df, columns='value'):
    d = {k: df[k].iloc[0] for k in ['country', 'habitat code', 'recorder', 'feature', 'timestamp']}
    d = d | {
        'mean': df[columns].mean(),
        'std': df[columns].std(),
        'skew': scipy.stats.skew(df[columns]),
        'kurtosis': scipy.stats.kurtosis(df[columns])
    }
    return pd.Series(d)


def main(infile=None, outfile=None, memory=24, cores=4, jobs=8, npartitions=None,
         is_long=False,
         # fps=12, window=60 * 12, hop=10 * 12,
         test=False, queue='test.short', binary=True, local=False, **kwargs):
    assert infile is not None
    assert outfile is not None

    outfile = Path(outfile)

    if not local:
        # Start cluster
        cluster = AltairGridEngineCluster(cores=cores, memory=memory, queue='test',
                                          name=None)  # .short', name=None)
        logging.info(cluster.job_script())
        cluster.scale(jobs=jobs)
        client = Client(cluster)
    else:
        memory_per_worker = "auto"
        if cores is not None and memory > 0:
            memory_per_worker = f'{memory / cores}GiB'

        client = Client(n_workers=cores,
                        threads_per_worker=1,
                        memory_limit=memory_per_worker)
        print(client)
        logging.info(client)

    # Read data
    df = dd.read_parquet(infile)
    logging.debug(f'Partitions: {df.npartitions}')
    if npartitions is not None:
        df = df.repartition(npartitions=npartitions)
    df = df.persist()

    logging.info('Initial Load')

    if is_long:
        ddf_long = df
    else:
        # Melt to long form
        metadata = df.iloc[:, :df.columns.get_loc('0')]
        features = df.iloc[:, df.columns.get_loc('0'):]

        ddf_long = df.melt(id_vars=metadata.columns, value_vars=features.columns,
                           var_name='frame', value_name='value')
        ddf_long = ddf_long.assign(idx=ddf_long.feature.astype(str) + ' ' + ddf_long.recorder.astype(
            str) + ' ' + ddf_long.timestamp.dt.date.astype(str))
        ddf_long = ddf_long.dropna(subset='value')
        ddf_long = ddf_long.set_index('idx')

    df_sstats = ddf_long.groupby('idx').apply(sstats).compute()
    # df_sstats.columns = [c[0] for c in df_sstats.columns[:-2]] + [c[1] for c in df_sstats.columns[-2:]]
    df_sstats.timestamp = df_sstats.timestamp.dt.date
    df_sstats = df_sstats.rename(columns={'timestamp': 'date'})

    logging.info(f'Outputting df to pandas parquet')
    df_sstats.to_parquet(outfile)


if __name__ == '__main__':
    parser = DaskArgumentParser('Compute astrograms from audio feature sets', memory=48, cores=1, jobs=95,
                                npartitions=None, queue='test.short')

    parser.add_argument('--long', dest='is_long', default=False, action='store_true')

    args = parser.parse_args()
    print(args)

    main(**vars(args))
