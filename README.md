# SoundADE
Acoustic Descriptor Extraction tool for processing sound on High Performance Computing clusters.

SoundADE will recursively identify audio files within a directory structure and read these for processing by the pipeline. The pipeline will then compute a set of acoustic descriptors, pull out BirdNET species detection probabilities. If site-level information such as latitude, longitude and timezone are provided as a separate sites file, these will be used to extract solar and weather data.

## Installation

### Python
Install `uv` package. You can then bundle dependencies. Finally you can run the test suite, and the different parts of the pipeline either independently or as a whole using the CLI.

```
uv sync --locked
uv run pytest -v
uv run main.py --help
```

### Docker
Install `docker-ce` package. You can then build and run the full pipeline using:

```sh
docker compose up --build
```

### Singularity
Some HPC admins don't give users sudo privileges required to run docker. The project can be built using singularity which doesn't require privileges and is usually installed on HPC systems.

```
singularity build --fakeroot app.sif app.def
```

When on a machine you have sudo privileges, fakeroot isn't required, you can run:
```sh
singularity build --ignore-fakeroot-command -F app.sif app.def
```

If using SLURM based HPC, scripts to schedule these builds can be found under the `slurm` directory.

As with docker, you must rebuild every time you make changes to the source code.

If you need to inspect your container once its build, run:

```sh
singularity exec --env-file .env --bind $DATA_PATH:/data app.sif /bin/sh
```

## Usage
Run the whole pipeline, specifying the relevant option depending on your setup configuration (docker / singularity / local python environment).

```
./run.sh
```

### Configuration
Create your own custom dataset config file (see examples in `./config`). This is where you should specify FFT and BirdNET parameters. Default parameters for the FFT and BirdNET will be set by the pipeline if none are specified, however you need to specify a means for the pipeline to extract information from the file paths, such as `timestamp` and site-level information in the form of a regular expression. You can specify additional site-level information, see [below](#location-information) for more details.

## Environment Variables
`.env` contains run time settings for the pipeline:

- `DATA_PATH`: the path to the audio data you want processed N.B. Your `site_locations.parquet` file must also be in this location
- `SAVE_PATH`: the path to where you want the results saved. Your `locations.parquet` file must be in this location. For more details on the `locations.parquet` see below.
- `CORES`: the number of cores (local) or jobs (HPC) to be deployed to process your data
- `MEM_PER_CPU`: the integer number of gigabytes of RAM deployed *per core or job*

### Location Information
A regular expression is required to find and extract audio files along with their location information. See `./config` for examples.

A file specifying site-specific information is required for the pipeline to run.

| site_id | site_name | latitude | longitude | timezone |
|---------|-----------|----------|-----------|----------|
| string \| integer | string | float32 | float32 | string |

You need to ensure the `site_name` field matches the regular expression defined in the data class.

For example, consider the following folder structure:

```
└─── <site_level_1>
    └── <site_level_2>
        ├── <site_level_2>
        ├── ├─ <timestamp>.wav
        │   ├─ ....
        │   └─ <timestamp>.wav
        ├── <site_level_2>
        └── ├─ <timestamp>.wav
            ├─ ....
            └─ <timestamp>.wav
```

The `site_name` variable should match `/<site_level_1>/<site_level_2>/<site_level_3>`.

```
└─── EC
    └── TE
        ├── 9
        ├── ├─ 20150619_0630.wav
        │   ├─ ....
        │   └─ 20150621_0317.wav
        ├── 10
        └── ├─ 20150619_0630.wav
            ├─ ....
            └─ 20150621_0317.wav
```

In this case `<site_level_1>` is the country (EC = Ecuador), `<site_level_2>` is a site identifier (TE), and `<site_level_3>` is a recorder ID number. Therefore the `site_name` column must contain records `'/EC/TE/9'` for `'/EC/TE/10'`. The depth for the site level is arbitrary, you can define as many as you like in the regular expression for discovering audio files.

## Tests
Run the test suite:

```
uv run pytest -v
```

## Development
If you want to make a contribution to the codebase, please correspond with the package creators specified in the `pyproject.toml`.
