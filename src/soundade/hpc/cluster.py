import os

from dask_jobqueue import SGECluster, SLURMCluster
from dask_jobqueue.core import Job
from distributed import Scheduler

# LOGPATH = (Path(__file__).parent / '..' / '..' / '..' / 'jobs').relative_to(Path('.').resolve())

QUEUE_WALLTIMES = {
    'test.short': f'{2 * 60 * 60}',
    'test': f'{8 * 60 * 60}',
    'test.long': f'680400',  # {7.875 * 24 * 60 * 60}
    'verylong': f'{30 * 24 * 60 * 60}'
}


# TODO reimplement as factory? (https://realpython.com/factory-method-python/)

# TODO make walltime an option
# Adapted from https://arctraining.github.io/swd6_hpp/05_parallelisation.html#dask-jobqueue
class ARCCluster(SGECluster):
    def __init__(self, cores=1, walltime="24:00:00", memory=1, queue=None, **job_kwargs):
        super().__init__(
            cores=cores,
            interface="ib0",
            walltime=walltime,
            memory=f"{memory} G",
            resource_spec=f"h_vmem={memory}G",
            scheduler_options={"dashboard_address": ":2727"},
            job_extra=[
                "-V",  # export all environment variables
                f"-pe smp {cores}",
            ],
            local_directory=os.sep.join([
                os.environ.get("PWD"),
                "dask-worker-space"]),
            **job_kwargs
        )


class ArtemisCluster(SLURMCluster):
    def __init__(self, n_workers=0, job_cls: Job = None, loop=None, security=None, shared_temp_directory=None,
                 silence_logs="error", name=None, asynchronous=False, dashboard_address=None, host=None,
                 scheduler_options=None, scheduler_cls=Scheduler, interface=None, protocol=None, config_name=None,
                 cores=None, memory=None, queue='verylong',  # project=None,
                 **job_kwargs):
        resource_spec = f'--mem={memory}G'
        total_memory = f'{memory}G'
        walltime = QUEUE_WALLTIMES[queue]
        job_extra_directives = [
            f'-p {queue}',
        ]
        if cores is not None and cores > 1:
            resource_spec = f'--mem={memory / cores}G'
            job_extra_directives.append(f'-n {cores}')

        super().__init__(n_workers, job_cls, loop, security, shared_temp_directory, silence_logs, name, asynchronous,
                         dashboard_address, host, scheduler_options, scheduler_cls, interface, protocol, config_name,
                         walltime=walltime, resource_spec=resource_spec, job_extra_directives=job_extra_directives,
                         cores=cores, memory=total_memory,
                         **job_kwargs)


class AltairGridEngineCluster(SGECluster):
    def __init__(self, n_workers=0, job_cls: Job = None, loop=None, security=None, shared_temp_directory=None,
                 silence_logs="error", name=None, asynchronous=False, dashboard_address=None, host=None,
                 scheduler_options=None, scheduler_cls=Scheduler, interface=None, protocol=None, config_name=None,
                 cores=None, memory=None, queue='verylong',  # project=None,
                 **job_kwargs):
        resource_spec = f'm_mem_free={memory}G'
        total_memory = f'{memory}GB'
        walltime = QUEUE_WALLTIMES[queue]
        job_extra_directives = [
            f'-jc {queue}',
            #   '-pe openmp 16',
        ]
        if cores is not None and cores > 1:
            resource_spec = f'm_mem_free={memory / cores}G'
            job_extra_directives.append(f'-pe openmp {cores}')

        # if name is not None:
        #     p = LOGPATH / name
        #     p = p.with_suffix('.log')
        #     p.parent.mkdir(parents=True, exist_ok=True)
        #     p.touch()
        #
        #     print(p)
        #
        #     job_extra_directives.append(f'-o {p}')

        super().__init__(n_workers, job_cls, loop, security, shared_temp_directory, silence_logs, name, asynchronous,
                         dashboard_address, host, scheduler_options, scheduler_cls, interface, protocol, config_name,
                         walltime=walltime, resource_spec=resource_spec, job_extra_directives=job_extra_directives,
                         cores=cores, memory=total_memory,
                         **job_kwargs)


clusters = {
    'arc': ARCCluster,
    'altair': AltairGridEngineCluster,
    'artemis': ArtemisCluster
}
