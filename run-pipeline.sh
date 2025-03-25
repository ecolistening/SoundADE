#!/bin/bash

export CODE_PATH=$HOME/SoundADE
cd $CODE_PATH

source $CODE_PATH/settings.env

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
           singularity run --env "CORES=$CORES" -B $DATA_PATH:/data $CODE_PATH/pipeline.sif
           ;;
        d) echo "Running using docker"
           sudo docker rm sa-pipeline
           sudo docker run --name sa-pipeline -e CORES=$CORES -p 8787:8787 -v $DATA_PATH:/data soundade
           sudo chown -R $USER:$USER $DATA_PATH  # Fix permissions on sudo written folders
           ;;
        b) echo "Running using slurm batch scheduler"
           sbatch --cpus-per-task=$CORES --mem-per-cpu=$MEM_PER_CPU $CODE_PATH/slurm/schedule-pipeline.sh
           ;;
        l) echo "Running in local anaconda environment"
           $CODE_PATH/pipeline-steps.sh -l
           ;;
        *) usage
    esac
done
