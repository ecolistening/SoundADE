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
    echo "Process files"
    xargs -a $PROFILE_PATH \
          $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/process_files.py \
          "--indir=$DATA_PATH" \
          "--outfile=$DATA_PATH/processed.parquet" \
          "--cores=$CORES" \
          "--local" \
          "--local_threads=1" \
          "--memory=$MEM_PER_CPU"
fi

if [[ ${STEPS_ARRAY[1]} = true ]] ; then
    echo "Append metadata"
    $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/append_metadata.py \
                                         "--infile=$DATA_PATH/processed.parquet" \
                                         "--outfile=$DATA_PATH/solar.parquet" \
                                         "--sitesfile=$DATA_PATH/site_locations.parquet" \
                                         "--local" \
                                         -F -C -T \
                                         "--cores=$CORES" \
                                         "--memory=$MEM_PER_CPU"
fi

if [[ ${STEPS_ARRAY[2]} = true ]] ; then
    echo "prefilter"
    $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/prefilter.py \
                                         "--infile=$DATA_PATH/solar.parquet" \
                                         "--outfile=$DATA_PATH/prefilter.parquet" \
                                         "--local" \
                                         "--cores=$CORES" \
                                         "--memory=$MEM_PER_CPU"
fi

if [[ ${STEPS_ARRAY[3]} = true ]] ; then
    echo "to long"
    $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/to_long.py \
                                         "--infile=$DATA_PATH/prefilter.parquet" \
                                         "--outfile=$DATA_PATH/to_long.parquet" \
                                         "--first-frame=3" \
                                         "--local" \
                                         "--cores=$CORES" \
                                         "--memory=$MEM_PER_CPU"
fi

if [[ ${STEPS_ARRAY[4]} = true ]] ; then
    echo "summary stats"
    $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/summary_stats.py \
                                         "--infile=$DATA_PATH/to_long.parquet" \
                                         "--outfile=$DATA_PATH/summary_stats.parquet" \
                                         "--long" \
                                         "--local" \
                                         "--cores=$CORES" \
                                         "--memory=$MEM_PER_CPU"
fi

if [[ ${STEPS_ARRAY[5]} = true ]] ; then
    echo "histograms"
    $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/astrograms.py \
                                         "--infile=$DATA_PATH/to_long.parquet" \
                                         "--outfile=$DATA_PATH/histograms.parquet" \
                                         "--histogram" \
                                         "--long" \
                                         "--local" \
                                         "--cores=$CORES" \
                                         "--memory=$MEM_PER_CPU"
fi

if [[ ${STEPS_ARRAY[6]} = true ]] ; then
    echo "astrograms"
    $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/astrograms.py \
                                         "--infile=$DATA_PATH/to_long.parquet" \
                                         "--outfile=$DATA_PATH/astrograms.parquet" \
                                         "--astrogram" \
                                         "--long" \
                                         "--local" \
                                         "--cores=$CORES" \
                                         "--memory=$MEM_PER_CPU"
fi

if [[ ${STEPS_ARRAY[7]} = true ]] ; then
    echo "lz complexity"
    $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/lempel_ziv_complexity.py \
                                         "--infile=$DATA_PATH/prefilter.parquet" \
                                         "--outfile=$DATA_PATH/lz_complexity.parquet" \
                                         "--first-frame=3" \
                                         "--local" \
                                         "--cores=$CORES" \
                                         "--memory=$MEM_PER_CPU"
fi

#Capture environment and git hash for reproducibility
conda run -n soundade conda env export -c conda-forge -c anaconda -c defaults > $DATA_PATH/run-environment/environment.yml

if [[ -n "$GIT_COMMIT" ]]; then
    echo $GIT_COMMIT > $DATA_PATH/run-environment/commit.hash;
else
    git rev-parse HEAD > $DATA_PATH/run-environment/commit.hash;
fi


