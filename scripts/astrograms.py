from pathlib import Path

import dask.dataframe as dd
from dask.distributed import Client, progress
import numpy as np

from soundade.stats.histogram import bin_and_cut, time_of_day_heatmap
from soundade.audio.binarisation import mean_threshold
from soundade.audio.lempel_ziv import lempel_ziv_complexity, lzc_row, windowed_lzc
from soundade.audio.symbolisation import bin_edges, symbolise
from soundade.hpc.cluster import AltairGridEngineCluster
from soundade.hpc.arguments import DaskArgumentParser

import logging
logging.basicConfig(level=logging.INFO)

def main(infile=None, outfile=None, memory=24, cores=4, jobs=8, npartitions=None,
         astrogram=True, is_long=False,
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
    print('df: (',len(df),',',len(df.columns),')')
    print(f'df.meta: {df._meta}')
    print(f'df.columns: {df.columns}')
    
    if is_long:
        df_long = df
    else:
        # Melt to long form
        metadata = df.iloc[:, :df.columns.get_loc('0')]
        features = df.iloc[:, df.columns.get_loc('0'):]
    
        df_long = df.melt(id_vars=metadata.columns, value_vars=features.columns,
                    var_name='frame', value_name='value')#.compute()
    
        # logging.info('Melted')
        # print(df_long._meta)
        # print(df_long.columns)
    
    df_long = df_long.assign(date=lambda r: r.timestamp.dt.date.astype('datetime64[ns]'))
    
        # logging.info('Date assignment')
        # print(df_long._meta)
        # print(df_long.columns)
    
    # Create astrograms
    nbins = 10
    meta = {'country': 'string',
            'habitat code': 'string',
            'recorder': 'i4',
            'date': 'datetime64[ns]',
            'dddn': 'string'} | dict([i, 'f8'] for i in range(nbins))

    if astrogram:
        groupby = ['country', 'habitat code', 'recorder', 'date', 'dddn']
    else:
        groupby = ['country', 'habitat code', 'recorder', 'date']
        meta.pop('dddn')

    df_astro = df_long.groupby('feature').apply(bin_and_cut, column='value',
                                                groupby=groupby,
                                                upper=0.99, lower=0.01, nbins=nbins, reset_index=True,
                                                meta=meta).rename(columns={i:str(i) for i in range(10)})
    # df_astro.loc[:,'0':] = df_astro.loc[:,'0':].div(df_astro.loc[:,'0':].sum(axis=1), axis=0)

    logging.info('Astrogram')
    print(df_astro._meta)
    print(df_astro.columns)
    
    logging.debug(df_astro.columns)
    
    # df_astro = df_astro.drop(columns='date')
    
    # Date columns might not work for this...
    # # Output
    # logging.info(f'Outputting ddf to parquet')
    # # Dask Parquet
    # dd.to_parquet(df_astro, outfile.with_stem(f'{outfile.stem}_dask'), version='2.6', allow_truncated_timestamps=True)
    # logging.info(f'Finished outputting dff to parquet')
    # Pandas Parquet
    logging.info(f'Outputting df to pandas parquet')
    df_pandas = df_astro.compute()
    df_pandas.to_parquet(outfile)

if __name__ == '__main__':
    parser = DaskArgumentParser('Compute astrograms from audio feature sets', memory=48, cores=1, jobs=95,
                                npartitions=None, queue='test.short')

    # parser.add_argument('--fps', type=int, default=12, help='Number of feature frames per second.')
    # parser.add_argument('--window', type=int, default=60 * 12, help='Number of feature frames for a lzc frame.')
    # parser.add_argument('--hop', type=int, default=10 * 12, help='Number of feature frames for the hop.')

    astro_histo = parser.add_mutually_exclusive_group()
    astro_histo.add_argument('--histogram', dest='astrogram', default=True, action='store_false')
    astro_histo.add_argument('--astrogram', dest='astrogram', default=True, action='store_true')
    
    parser.add_argument('--long', dest='is_long', default=False, action='store_true')

    args = parser.parse_args()
    print(args)

    main(**vars(args))
