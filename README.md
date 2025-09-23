# SoundADE
Acoustic Descriptor Extraction tool for processing sound on High Performance Computing clusters

## Installation

### Docker
Docker caches builds in layers and so is convenient for fast iteration and development when running locally. Using the `docker-ce` package, you can build a docker image using:

`sudo docker build --tag 'soundade' .`

You must redo this every time you make changes to the source code. However `settings.env` is passed in at run time so you do not need to rebuild if you're just changing things in there.

### Singularity/apptainer
Some HPC admins don't give users the sudo priveleges required to run docker. Therefore the project can be built using singularity/apptainer which doesn't require priveleges and is usually installed on HPC systems.

`singularity build --ignore-fakeroot-command -F app.sif app.def`

If using SLURM based HPC, scripts to schedule these builds can be found under the `slurm` directory.

As with docker, you must rebuild every time you make changes to the source code. However `settings.env` is still passed in at run time so you do not need to rebuild if you're just changing things in there.

### Local environment
You can use a local conda environment if you want. Be aware that your local conda version/API might have drifted since the development of this project.

`conda env create -f environment.yml`

Then add the source code to the conda environment:
`conda run -n soundade python -m pip install .`


## Running the code

`settings.env` contains run time settings for the pipeline:

- `DATA_PATH`: the path to the audio data you want processed N.B. Your `site_locations.parquet` file must also be in this location
- `PROFILE_PATH`: the path to a text file defining FFT parameters and dataset options N.B. these parameters are in a separate file to `settings.env` on purpose, so that they can be captured as part of the run-time environment record
- `CORES`: the number of cores (local) or jobs (HPC) to be deployed to process your data
- `MEM_PER_CPU`: the integer number of gigabytes of RAM deployed *per core or job*
- `STEPS`: a colon`:` delimeted string of `true` and `false` variables denoting which steps of the pipeline to run

Depending on your installation method you can use the `run-pipeline.sh` script with the appropriate flag:

```
usage: ./run-pipeline.sh -s # Run using singularity container
       ./run-pipeline.sh -d # Run using docker container
       ./run-pipeline.sh -b # Run using slurm batch scheduler
       ./run-pipeline.sh -l # Run in local anaconda environment
```

Output parquet files will be written to `$DATA_PATH/processed`, and details of the runtime settings/environment will be written to `$DATA_PATH/run-environment`. These runtime settings might help to reconstruct pipeline settings to aid reproducibility.

### Troubleshooting

#### Solving Environment | Killed
Sometimes the environement solving step can take up too much memory (especially for a login node on a cluster). This can be fixed by removing channels from the YAML file. For me, removing `anaconda` and `defaults` did the trick.
