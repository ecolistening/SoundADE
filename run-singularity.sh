#!/bin/bash
export CODE_PATH=$HOME/dev/SoundADE
export DATA_PATH=$HOME/dev/SoundADE/data/eco1
export PROFILE_PATH=$HOME/dev/SoundADE/profiles/singularity/single

mkdir -p $DATA_PATH/processed
mkdir -p $DATA_PATH/run-environment
cp $PROFILE_PATH $DATA_PATH/run-environment/profile
singularity run -B $DATA_PATH:/data pipeline.sif
