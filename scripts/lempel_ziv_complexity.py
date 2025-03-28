from pathlib import Path

import dask.dataframe as dd
from dask.distributed import Client, progress
import numpy as np

from soundade.audio.binarisation import mean_threshold
from soundade.audio.lempel_ziv import lempel_ziv_complexity, lzc_row
from soundade.audio.symbolisation import bin_edges, symbolise
from soundade.hpc.cluster import AltairGridEngineCluster
from soundade.hpc.arguments import DaskArgumentParser


# meta_cols = ['path', 'file', 'country', 'location', 'habitat', 'recorder', 'channel', 'timestamp', 'feature']

def binarised(df, outfile, first_frame='0'):
    # Compute lempel ziv for each binarisations
    # binarisations = {
    #     'mean threshold': mean_threshold,
    #     # 'median threshold': median_threshold,
    #     # 'kmeans': kmeans
    # }
    # lz_cols = [b for b in binarisations]
    lzc = lambda s: lempel_ziv_complexity(s, normalisation='proportional')
    # meta = dict([(b, 'f8') for b in binarisations])
    _last_col = df.columns[-1]
    # df[lz_cols] = lzcs_row, axis=1, result_type='expand', meta=meta,
    #                                     binarisations=binarisations, complexity=lzc)

    df['lempel ziv complexity'] = df.loc[:, first_frame:].apply(lzc_row, axis=1, binarisation=mean_threshold,
                                                                complexity=lzc)

    df = df.drop(columns=df.loc[:, first_frame:_last_col])

    # dd.to_parquet(df, outfile.with_stem(f'{outfile.stem}_meta'), version='2.6', allow_truncated_timestamps=True)
    # print(f'Finished outputting df_meta to parquet')
    # id_vars = [c for c in df.columns if c in set(df.columns) - set()]  # Orders ID columns correctly
    # df_long = df.melt(id_vars=id_vars, value_vars=lz_cols, var_name='binarisation',
    #                   value_name='lempel ziv complexity')  # .persist()
    # print(f'df_long columns: {df_long.columns} | paritions: {df_long.npartitions}')
    # progress(df_long)
    print(f'Outputting df to parquet')
    # Dask Parquet
    dd.to_parquet(df, outfile.with_stem(f'{outfile.stem}_dask'), version='2.6', allow_truncated_timestamps=True)
    print(f'Finished outputting df to parquet')
    # Pandas Parquet
    df_pandas = df.compute()
    df_pandas.to_parquet(outfile)


def symbolised(df, outfile, n_symbols, groupby='feature', first_frame='0'):
    colname = 'value'
    lzc = lambda s: lempel_ziv_complexity(s, normalisation='random', n_symbols=n_symbols)

    # Compute the bins for each feature (or other group, selectable by the groupby parameter)
    bin_edges = df.groupby(groupby).apply(lambda d: bin_edges(d.loc[:, first_frame:].dropna(axis=1), bins=n_symbols))

    # Compute the lzc for each row
    _last_col = df.columns[-1]
    df['binarisation'] = f'symbolised_{n_symbols}'
    df[colname] = df.loc[:, first_frame:].apply(
        lambda r: lempel_ziv_complexity(symbolise(r, bin_edges[r[groupby]]), n_symbols=n_symbols),
        axis=1, result_type='expand', meta={colname: 'f8'})
    df = df.drop(columns=df.loc[:, first_frame:_last_col])  # Drop the raw data

    # Output the data to a dask parquet
    dd.to_parquet(df, outfile.with_stem(f'{outfile.stem}_meta'), version='2.6', allow_truncated_timestamps=True)
    print(f'Finished outputting to dask parquet')

    df_pandas = df.compute()
    df_pandas.to_parquet(outfile)


def main(infile=None, outfile=None, memory=24, cores=4, jobs=8, npartitions=None,
         test=False, queue='test.short', binary=True, local=False, first_frame='0', **kwargs):
    assert infile is not None
    assert outfile is not None

    outfile = Path(outfile)

    if not local:
        # Start cluster
        cluster = AltairGridEngineCluster(cores=cores, memory=memory, queue='test.long',
                                          name=None)  # .short', name=None)
        print(cluster.job_script())
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

    # Read data
    df = dd.read_parquet(infile)
    print(f'Partitions: {df.npartitions}')
    if npartitions is not None:
        df = df.repartition(npartitions=npartitions)
    df = df.persist()

    if binary:
        print('Computing binarised LZ')
        binarised(df, outfile, first_frame=first_frame)


if __name__ == '__main__':
    parser = DaskArgumentParser('Compute Lempel Ziv Complexity from audio feature sets', memory=48, cores=1, jobs=8,
                                npartitions=None, queue='test.short')

    symbols = parser.add_mutually_exclusive_group()
    symbols.add_argument('--binary', dest='binary', default=True, action='store_false')
    symbols.add_argument('--symbols', dest='binary', default=True, action='store_true')

    parser.add_argument('--first-frame', default='0', type=str,
                        help='First frame')

    args = parser.parse_args()
    print(args)

    main(**vars(args))
