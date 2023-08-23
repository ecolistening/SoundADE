from pathlib import Path

from dask import config as cfg
from dask.distributed import Client

from soundade.data.datasets import datasets, Dataset
from soundade.hpc.arguments import DaskArgumentParser
from soundade.hpc.cluster import clusters

cfg.set({'distributed.scheduler.worker-ttl': None})

defaults = {
    'memory': 32,
    'cores': 8,
    'jobs': 12,
    'frame': 16000,  # frames
    'hop': 4000,  # frames
    'n_fft': 16000,  # frames
    'npartitions': 2000
}


def main(cluster=None, indir=None, outfile=None, memory=64, cores=4, jobs=2,
         dataset=None, frame=0, hop=0, n_fft=0, npartitions=None,
         local=False, save_preprocessed=None, compute=False,
         **kwargs):
    if not local:
        # Start cluster
        cluster = clusters[cluster](cores=cores, memory=memory, queue='test.long',
                                    name=None)  # .short', name=None)
        print(cluster.job_script())
        cluster.scale(jobs=jobs)
        client = Client(cluster)
    else:
        client = Client(n_workers=1, threads_per_worker=1)
        print(client)

    outfile = Path(outfile)

    ds: Dataset = datasets[dataset]

    b = ds.load(indir, npartitions=npartitions)

    if save_preprocessed is not None:
        Path(save_preprocessed).mkdir(parents=True, exist_ok=True)

    b = ds.preprocess(b, save=save_preprocessed)

    # Extract all of the features
    b = ds.extract_features(b, frame, hop, n_fft).persist()

    # Convert to dataframe format
    ddf = ds.to_dataframe(b)

    # Repartition and extract metadata
    ddf = ddf.repartition(npartitions=npartitions).persist()
    ddf = ds.metadata(ddf)

    # ddf.visualize(filename=f'{datetime.now()}-tasks.svg')
    # ddf.visualize(filename=f'{datetime.now()}-tasks-optimised.svg', optimize_graph=True)

    ddf = ddf.repartition(npartitions=npartitions).persist()

    # print(ddf.columns)

    ds.to_parquet(ddf, path=outfile)
    # except ValueError as e:
    #     print(e)
    #     print('Schema: ', ddf.schema)
    #     print('Columns: ', ddf.columns)

    if compute:
        ds.to_parquet(ddf, path=outfile.with_stem(f'{outfile.stem}_computed'), compute=True)


if __name__ == '__main__':
    parser = DaskArgumentParser('Extract features from audio files')

    parser.add_argument('--indir', default=None, help='Folder containing input files.')

    parser.add_argument('--dataset', type=str, help='Which dataset to use')

    parser.add_argument('--frame', type=int, help='Number of audio frames for a feature frame.')
    parser.add_argument('--hop', type=int, help='Number of audio frames for the hop.')
    parser.add_argument('--n_fft', type=int, help='Number of audio frames for the n_fft.')

    parser.add_argument('--save-preprocessed', default=None, help='Save the preprocessed files to directory.')
    parser.add_argument('--compute', default=False, action='store_true',
                        help='Compute the pandas dataframe and save to parquet.')

    parser.set_defaults(**defaults)

    # TODO test that outfile is not None

    args = parser.parse_args()

    print(args)

    main(**vars(args))
