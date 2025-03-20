#!/bin/bash
cd $CODE_PATH

export CODE_PATH=$HOME/SoundADE
export DATA_PATH=$HOME/soundade-data/eco1
export PROFILE_PATH=$HOME/SoundADE/profiles/single

mkdir -p $DATA_PATH/processed
mkdir -p $DATA_PATH/run-environment
cp $PROFILE_PATH $DATA_PATH/run-environment/profile

usage()
{
    echo "usage: $0 -c # Run using container"
    echo "       $0 -l # Run in local anaconda environment"
}

while getopts "cl" flag; do
    case ${flag} in
        c) echo "Running using container"
           singularity run -B $DATA_PATH:/data pipeline.sif
           ;;
        l) echo "Running in local anaconda environment"
           ./pipeline-steps.sh -l
           ;;
        *) usage
    esac
done
