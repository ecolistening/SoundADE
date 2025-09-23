import argparse
from pathlib import Path

def parser(description, **kwargs):
    parser = argparse.ArgumentParser(description=description, **kwargs)

    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument('--ecuador', action='store_true', help='Process the files from Ecuador')
    dataset_group.add_argument('--uk', action='store_true', help='Process the files from the UK')

    data_location_group = parser.add_mutually_exclusive_group()
    data_location_group.add_argument('--server', action='store_true', help='Use server paths')
    data_location_group.add_argument('--local', action='store_true', help='Use local paths')

    parser.add_argument('--serial', action='store_true', help='Run in serial (i.e. do not parallelise using joblib')

    parser.add_argument('--filtered', action='store_true', help='Use the filtered dataset in the analysis.')

    parser.add_argument('--test', action='store_true', help='Test run. Limit to first 10 items.')

    return parser


class DaskArgumentParser(argparse.ArgumentParser):
    def __init__(self, description, memory=128, cores=1, jobs=4, npartitions=None, queue='general', **kwargs) -> None:
        super().__init__(description=description, **kwargs)
        self.add_argument('--cluster', default='artemis', help='Which cluster to use?')

        self.add_argument('--infile', type=lambda p: Path(p).expanduser(), default=None, help='Input parquet file(s).')
        self.add_argument('--outfile', type=lambda p: Path(p).expanduser(), default=None, help='Parquet file to save results.')

        self.add_argument('--memory', default=memory, type=int,
                          help='Amount of memory required in GB (total per node).')
        self.add_argument('--cores', default=cores, type=int, help='Number of cores per node.')
        self.add_argument('--jobs', default=jobs, type=int, help='Number of simultaneous jobs.')
        self.add_argument('--npartitions', default=npartitions, type=int,
                          help='Number of dask partitions for the data.')
        self.add_argument('--queue', default=queue, type=str, help='Job queue to select.')

        local = self.add_mutually_exclusive_group()
        local.add_argument('--local', dest='local', default=True, action='store_true')
        local.add_argument('--hpc', dest='local', default=False, action='store_false')
