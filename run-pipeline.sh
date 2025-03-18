#!/bin/bash
export PYTHONPATH="$BASE_PATH/src:$PYTHONPATH"
cd $BASE_PATH
conda run -n soundade conda env export -c conda-forge -c anaconda -c defaults > $BASE_PATH/../data/environments/environment-current.yml
conda run -n soundade xargs -a $PROFILE_PATH python $BASE_PATH/scripts/process_files.py
