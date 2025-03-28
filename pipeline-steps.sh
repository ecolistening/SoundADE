#!/bin/bash
CONDA_PATH=/opt/conda

IFS=':' read -ra STEPS_ARRAY <<< "$STEPS"

# Option to use local conda environments
while getopts "l" flag; do
    case ${flag} in
        l) CONDA_PATH=$HOME/.conda
           ;;
    esac
done

cd $CODE_PATH
if [[ ${STEPS_ARRAY[0]} = true ]] ; then
    xargs -a $PROFILE_PATH \
          $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/process_files.py \
          "--indir=$DATA_PATH" \
          "--outfile=$DATA_PATH/processed" \
          "--cores=$CORES" \
          "--local" \
          "--local_threads=1" \
          "--memory=0"
fi

if [[ ${STEPS_ARRAY[1]} = true ]] ; then
    $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/append_metadata.py \
                                         "--infile=$DATA_PATH/processed" \
                                         "--outfile=$DATA_PATH/solar" \
                                         "--sitesfile=$DATA_PATH/site_locations.parquet" \
                                         -F -C -T \
                                         "--cores=$CORES" \
                                         "--memory=0"
fi

if [[ ${STEPS_ARRAY[2]} = true ]] ; then
    $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/prefilter.py \
                                         "--infile=$DATA_PATH/solar" \
                                         "--outfile=$DATA_PATH/pre" \
                                         "--local" \
                                         "--cores=$CORES" \
                                         "--memory=0"
fi

if [[ ${STEPS_ARRAY[3]} = true ]] ; then
    $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/to_long.py \
                                         "--infile=$DATA_PATH/pre" \
                                         "--outfile=$DATA_PATH/to_long" \
                                         "--first-frame=3" \
                                         "--local" \
                                         "--cores=$CORES" \
                                         "--memory=0"
fi

if [[ ${STEPS_ARRAY[4]} = true ]] ; then
    $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/summary_stats.py \
                                         "--infile=$DATA_PATH/to_long" \
                                         "--outfile=$DATA_PATH/summary_stats" \
                                         "--long" \
                                         "--local" \
                                         "--cores=$CORES" \
                                         "--memory=0"
fi

if [[ ${STEPS_ARRAY[5]} = true ]] ; then
    $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/astrograms.py \
                                         "--infile=$DATA_PATH/to_long" \
                                         "--outfile=$DATA_PATH/histograms" \
                                         "--histogram" \
                                         "--long" \
                                         "--local" \
                                         "--cores=$CORES" \
                                         "--memory=0"
fi

if [[ ${STEPS_ARRAY[6]} = true ]] ; then
    $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/astrograms.py \
                                         "--infile=$DATA_PATH/to_long" \
                                         "--outfile=$DATA_PATH/astrograms" \
                                         "--astrogram" \
                                         "--long" \
                                         "--local" \
                                         "--cores=$CORES" \
                                         "--memory=0"
fi

#Capture environment and git hash for reproducibility
conda run -n soundade conda env export -c conda-forge -c anaconda -c defaults > $DATA_PATH/run-environment/environment.yml

if [[ -n "$GIT_COMMIT" ]]; then
    echo $GIT_COMMIT > $DATA_PATH/run-environment/commit.hash;
else
    git rev-parse HEAD > $DATA_PATH/run-environment/commit.hash;
fi


