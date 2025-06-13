#!/bin/bash
#SBATCH --job-name=soundade-pipeline
#SBATCH --output=soundade-pipeline."%j".out
#SBATCH --error=soundade-pipeline."%j".err
#SBATCH --mail-user $USER@sussex.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --partition=general

set -a
source settings.env
set +a

singularity run --env "CORES=$CORES" --env "MEM_PER_CPU=$MEM_PER_CPU" --env "STEPS=$STEPS" -B $DATA_PATH:/data pipeline.sif
