# SoundADE
Acoustic Descriptor Extraction tool for processing sound on High Performance Computing clusters

## Installation

### Docker
Install `docker-ce` package. You can then build and run the full pipeline using:

```sh
docker compose up --build
```

### Singularity
Some HPC admins don't give users sudo privileges required to run docker. The project can be built using singularity which doesn't require privileges and is usually installed on HPC systems.

```sh
singularity build --ignore-fakeroot-command -F app.sif app.def
```

If using SLURM based HPC, scripts to schedule these builds can be found under the `slurm` directory.

As with docker, you must rebuild every time you make changes to the source code.

## Usage
Run the whole pipeline, specifying the relevant option depending on your setup configuration (docker / singularity / local python environment).

```
./run.sh
```

## Environment Variables
`.env` contains run time settings for the pipeline:

- `DATA_PATH`: the path to the audio data you want processed N.B. Your `site_locations.parquet` file must also be in this location
- `SAVE_PATH`: the path to where you want the results saved. Your `locations.parquet` file must be in this location. For more details on the `locations.parquet` see below.
- `CORES`: the number of cores (local) or jobs (HPC) to be deployed to process your data
- `MEM_PER_CPU`: the integer number of gigabytes of RAM deployed *per core or job*
- `DATASET`: must map to the name of a dataset class in `src/soundade/datasets`.
- `SAMPLE_RATE`: a resample rate for the audio. If comparable features are desired across different sample rates, you should resample to the minimum sample rate in your dataset.
- `FRAME`: the number of samples used for computing a frame for certain audio features e.g. zero crossing rate.
- `HOP`: the hop size used for the FFT for audio features derived from the spectrogram, e.g. spectral centroid.
- `N_FFT`: the number of samples per FFT window, determining the time-frequency resolution for audio features derived from the spectrogram.
- `MIN_CONF`: a threshold value for BirdNET species detection probabilities. Detections below this threshold are ignored. Defaults to 0.0.
- `SEGMENT_LEN`: for longer audio files, we first chunk the audio into segments. Defaults to 60s. Features are then calculated over each segment independently.

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

