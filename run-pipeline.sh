#!/bin/bash

export CODE_PATH=$HOME/SoundADE
cd $CODE_PATH

source $CODE_PATH/settings.env

usage()
{
    echo "usage: $0 -s # Run using singularity container"
    echo "       $0 -d # Run using docker container"
    echo "       $0 -b # Run using slurm batch scheduler"
    echo "       $0 -l # Run in local anaconda environment"
}

while getopts "sdblmp:" flag; do
    case ${flag} in
        s) DO_SINGULARITY=true
           ;;
        d) DO_DOCKER=true
           ;;
        b) DO_SLURM=true
           ;;
        l) DO_LOCAL=true
           ;;
        *) usage
    esac
done

for DIR in "${SUB_DIRS[@]}"; do
    export DATA_PATH=$DATA_ROOT/$DIR
    echo "Running pipeline for: $DATA_PATH"
    mkdir -p $DATA_PATH/processed/run-environment
    cp $PROFILE_PATH $DATA_PATH/processed/run-environment/profile.env
    cp $CODE_PATH/settings.env $DATA_PATH/processed/run-environment/settings.env

    if [[ $DO_SINGULARITY = true ]] ; then
        echo "Running using singularity"
        singularity run --env "CORES=$CORES" --env "MEM_PER_CPU=$MEM_PER_CPU" --env "STEPS=$STEPS" -B $DATA_PATH:/data $CODE_PATH/pipeline.sif
    fi

    if [[ $DO_DOCKER = true ]] ; then
        echo "Running using docker"
        sudo docker rm sa-pipeline
        sudo docker run --name sa-pipeline -e CORES=$CORES -e MEM_PER_CPU=$MEM_PER_CPU -e STEPS=$STEPS -p 8787:8787 -v $DATA_PATH:/data soundade
        sudo chown -R $USER:$USER $DATA_PATH  # Fix permissions on sudo written folders
    fi

    if [[ $DO_SLURM = true ]] ; then
        echo "Running using slurm batch scheduler"
        sbatch --cpus-per-task=$CORES --mem-per-cpu="${MEM_PER_CPU}G" $CODE_PATH/slurm/schedule-pipeline.sh
    fi

    if [[ $DO_LOCAL = true ]] ; then
        echo "Running in local anaconda environment"
        $CODE_PATH/pipeline-steps.sh -l
    fi
done
