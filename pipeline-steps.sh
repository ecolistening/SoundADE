#!/bin/bash

CONDA_PATH=/opt/conda

# Option to use local conda environments
while getopts "l" flag; do
    case ${flag} in
        l) CONDA_PATH=$HOME/.conda
           ;;
    esac
done

cd $CODE_PATH
xargs -a $PROFILE_PATH $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/process_files.py "--indir=$DATA_PATH" "--outfile=$DATA_PATH/processed" "--cores=$CORES"

#Capture environment and git hash for reproducibility
conda run -n soundade conda env export -c conda-forge -c anaconda -c defaults > $DATA_PATH/run-environment/environment.yml

if [ -n "$GIT_COMMIT" ]; then
    echo $GIT_COMMIT > $DATA_PATH/run-environment/commit.hash;
else
    git rev-parse HEAD > $DATA_PATH/run-environment/commit.hash;
fi


