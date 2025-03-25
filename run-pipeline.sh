#!/bin/bash

# Memory specification is only used to request resources from slurm;
# the actual pipeline uses all memory that is available
export MEM_PER_CPU=2G 
# Cores value is used by slurm and the pipeline
export CORES=1

# Profile contains FFT and data-set settings
export PROFILE_PATH=$HOME/SoundADE/profiles/single

# Location of your code and data
export CODE_PATH=$HOME/SoundADE
export DATA_PATH=$HOME/soundade-data/eco1

mkdir -p $DATA_PATH/processed
mkdir -p $DATA_PATH/run-environment
cp $PROFILE_PATH $DATA_PATH/run-environment/profile

usage()
{
    echo "usage: $0 -s # Run using singularity container"
    echo "       $0 -d # Run using docker container"
    echo "       $0 -b # Run using slurm batch scheduler"
    echo "       $0 -l # Run in local anaconda environment"
}

while getopts "sdbl" flag; do
    case ${flag} in
        s) echo "Running using singularity"
           singularity run --env "CORES=$CORES" -B $DATA_PATH:/data pipeline.sif
           ;;
        d) echo "Running using docker"
           sudo docker run --name sa-pipeline -e CORES=$CORES -p 8787:8787 -v $DATA_PATH:/data soundade
           sudo chown -R $USER:$USER $DATA_PATH  # Fix permissions on sudo written folders
           ;;
        b) echo "Running using slurm batch scheduler"
           sbatch --cpus-per-task=$CORES --mem-per-cpu=$MEM_PER_CPU $CODE_PATH/slurm/schedule-pipeline.sh
           ;;
        l) echo "Running in local anaconda environment"
           ./pipeline-steps.sh -l
           ;;
        *) usage
    esac
done
