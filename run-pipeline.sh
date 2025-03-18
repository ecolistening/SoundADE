#!/bin/bash
export PYTHONPATH="$BASE_PATH/src:$PYTHONPATH"
cd $BASE_PATH
conda run -n soundade xargs -a $PROFILE_PATH python $BASE_PATH/scripts/process_files.py
