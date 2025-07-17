#!/bin/bash
#SBATCH -J soundade-pipeline
#SBATCH -o soundade-pipeline."%j".out
#SBATCH -e soundade-pipeline."%j".err
#Default in slurm
#SBATCH --mail-user $USER@sussex.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -p general

cd $HOME/SoundADE
source ./settings.env
singularity run --env "CORES=$CORES" --env "MEM_PER_CPU=$MEM_PER_CPU" --env "STEPS=$STEPS" -B $DATA_PATH:/data $CODE_PATH/pipeline.sif
