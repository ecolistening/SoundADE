#!/bin/bash

export CODE_PATH=$HOME/SoundADE
export DATA_PATH=$HOME/soundade-data/eco1
export PROFILE_PATH=$HOME/SoundADE/profiles/single

mkdir -p $DATA_PATH/processed
mkdir -p $DATA_PATH/run-environment
cp $PROFILE_PATH $DATA_PATH/run-environment/profile

usage()
{
    echo "usage: $0 -s # Run using singularity container"
    echo "       $0 -d # Run using docker container"
    echo "       $0 -l # Run in local anaconda environment"
}

while getopts "sdl" flag; do
    case ${flag} in
        s) echo "Running using singularity"
           singularity run -B $DATA_PATH:/data pipeline.sif
           ;;
        d) echo "Running using docker"
           sudo docker run --name sa-pipeline --detach -p 8787:8787 -v $DATA_PATH:/data soundade
           ;;
        l) echo "Running in local anaconda environment"
           ./pipeline-steps.sh -l
           ;;
        *) usage
    esac
done
