import time
from pathlib import Path

from dask import config as cfg
from dask.distributed import Client

from soundade.data.datasets import datasets, Dataset
from soundade.hpc.arguments import DaskArgumentParser
from soundade.hpc.cluster import clusters

cfg.set({'distributed.scheduler.worker-ttl': None})

# Defaults are set for running in the container
defaults = {
    'indir': "/data",
    'outfile': "/data/processed",
    'local': True,
    'local_threads': 1,
    'memory': 0,
    'cores': 1,
    'frame': 2048,
    'hop': 512,
    'n_fft': 2048,
}


def main(cluster=None, indir=None, outfile=None, memory=0, cores=0, jobs=0,
         queue='general', dataset=None, frame=0, hop=0, n_fft=0, npartitions=None,
         local=True, local_threads=1, compute=False, debug=False,
         overwrite_parquet=True, save_preprocessed=None, 
         **kwargs):
    """
    Process audio files using the specified parameters.

    Args:
        cluster (str, optional): Name of the cluster to use. 'arc' or 'altair' or None if local==True. Defaults to None.
        indir (str, optional): Input directory containing audio files. Required.
        outfile (str, optional): Output file path. Required.
        memory (int, optional): Memory limit for each worker in GB. Defaults to 32.
        cores (int, optional): Number of CPU cores per worker. Defaults to 8.
        jobs (int, optional): Number of worker jobs to start. Defaults to 12.
        dataset (str, optional): Name of the dataset to use. One of the dictionary keys in soundade.data.datasets.datasets. Required.
        frame (int, optional): Frame size for feature extraction. Defaults to 16000.
        hop (int, optional): Hop size for feature extraction. Defaults to 4000.
        n_fft (int, optional): Number of FFT points for feature extraction. Defaults to 16000.
        npartitions (int, optional): Number of partitions for Dask DataFrame. Defaults to 2000.
        local (bool, optional): Flag indicating whether to run locally. Defaults to False.
        save_preprocessed (str, optional): Directory to save preprocessed data. Defaults to None.
        compute (bool, optional): Flag indicating whether to compute to a pandas dataframe and save the results as a final step. Defaults to False.

    Returns:
        None

    Raises:
        ValueError: If an error occurs during processing.

    Examples:
        >>> main(indir='./data/ecolistening', outfile='./data/processed/ecolistening/features.parquet',
        ...      dataset='SoundingOutDiurnal', frame=2048, hop=512, n_fft=1024,
        ...      local=True, save_preprocessed='./data/processed/ecolistening/processed_audio', compute=True)
        <Client: ...
    """
    t0 = time.time()
    if not local:
        # Start cluster
        cluster = clusters[cluster](cores=cores, memory=memory,
                                    queue=queue, name=None)
        print(cluster.job_script())
        cluster.scale(jobs=jobs)
        client = Client(cluster)
    else:
        if debug:
            cfg.set(scheduler='synchronous')

        memory_per_worker = f'{memory}GiB'

        client = Client(n_workers=cores,
                        threads_per_worker=local_threads,
                        memory_limit=memory_per_worker)
        print(client)

    outfile = Path(outfile)

    ds: Dataset = datasets[dataset]

    b = ds.load(indir, npartitions=npartitions)

    if save_preprocessed is not None:
        Path(save_preprocessed).mkdir(parents=True, exist_ok=True)

    #b = ds.preprocess(b, save=save_preprocessed)

    # Extract all of the features
    b = ds.extract_features(b, frame, hop, n_fft)

    # Convert to dataframe format
    ddf = ds.to_dataframe(b)

    # Extract metadata
    ddf = ds.metadata(ddf)

    ds.to_parquet(ddf, path=outfile, overwrite=overwrite_parquet)

    if compute:
        ds.to_parquet(ddf, path=outfile.with_stem(f'{outfile.stem}_computed'), compute=True)

    t1 = time.time()
    print(f'Time taken: {t1 - t0} seconds')


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

    parser.add_argument('--debug', default=False, action='store_true', help='Sets single-threaded for debugging.')
    parser.add_argument('--local_threads', type=int, default=1, help='Threads to use when run locally')
    parser.add_argument('--overwrite_parquet', default=True, action='store_true', help='Overwrites previous output data')

    parser.set_defaults(**defaults)

    # TODO test that outfile is not None

    args = parser.parse_args()

    print(args)

    main(**vars(args))
