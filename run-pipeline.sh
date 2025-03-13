#!/bin/bash
export PROFILE="single"
export BASE_PATH="$HOME/SoundADE"
export PROFILE_PATH="$BASE_PATH/profiles/artemis/$PROFILE"
export PYTHONPATH="$BASE_PATH/src:$PYTHONPATH"
cd $BASE_PATH
conda run -n soundade xargs -a $PROFILE_PATH python $BASE_PATH/scripts/process_files.py
