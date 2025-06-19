#!/bin/bash

# IFS=':' read -ra STEPS_ARRAY <<< "$STEPS"

# if [[ ${STEPS_ARRAY[0]} = true ]] ; then
xargs -a $PROFILE_PATH \
    python ./scripts/process_files.py \
    "--indir=$DATA_PATH" \
    "--outfile=$DATA_PATH/processed/processed.parquet" \
    "--cores=$CORES" \
    "--local" \
    "--local_threads=1" \
    "--memory=$MEM_PER_CPU"
# fi

# if [[ ${STEPS_ARRAY[1]} = true ]] ; then
# echo "Append metadata"
xargs -a $PROFILE_PATH \
    python ./scripts/append_metadata.py \
    "--infile=$DATA_PATH/processed/processed.parquet" \
    "--outfile=$DATA_PATH/processed/solar.parquet" \
    "--sitesfile=$DATA_PATH/site_locations.parquet" \
    "--local" \
    -F -C -T \
    "--cores=$CORES" \
    "--memory=$MEM_PER_CPU"
# fi

# if [[ ${STEPS_ARRAY[2]} = true ]] ; then
#     echo "prefilter"
#     python ./scripts/prefilter.py \
#         "--infile=$DATA_PATH/processed/solar.parquet" \
#         "--outfile=$DATA_PATH/processed/prefilter.parquet" \
#         "--local" \
#         "--cores=$CORES" \
#         "--memory=$MEM_PER_CPU"
# fi

# if [[ ${STEPS_ARRAY[3]} = true ]] ; then
#     echo "to long"
#     python ./scripts/to_long.py \
#         "--infile=$DATA_PATH/processed/prefilter.parquet" \
#         "--outfile=$DATA_PATH/processed/to_long.parquet" \
#         "--first-frame=3" \
#         "--local" \
#         "--cores=$CORES" \
#         "--memory=$MEM_PER_CPU"
# fi

# if [[ ${STEPS_ARRAY[4]} = true ]] ; then
#     echo "summary stats"
#     python ./scripts/summary_stats.py \
#         "--infile=$DATA_PATH/processed/to_long.parquet" \
#         "--outfile=$DATA_PATH/processed/summary_stats.parquet" \
#         "--long" \
#         "--local" \
#         "--cores=$CORES" \
#         "--memory=$MEM_PER_CPU"
# fi

# if [[ ${STEPS_ARRAY[5]} = true ]] ; then
#     echo "histograms"
#     python ./scripts/astrograms.py \
#         "--infile=$DATA_PATH/processed/to_long.parquet" \
#         "--outfile=$DATA_PATH/processed/histograms.parquet" \
#         "--histogram" \
#         "--long" \
#         "--local" \
#         "--cores=$CORES" \
#         "--memory=$MEM_PER_CPU"
# fi

# if [[ ${STEPS_ARRAY[6]} = true ]] ; then
#     echo "astrograms"
#     python ./scripts/astrograms.py \
#         "--infile=$DATA_PATH/processed/to_long.parquet" \
#         "--outfile=$DATA_PATH/processed/astrograms.parquet" \
#         "--astrogram" \
#         "--long" \
#         "--local" \
#         "--cores=$CORES" \
#         "--memory=$MEM_PER_CPU"
# fi

# if [[ ${STEPS_ARRAY[7]} = true ]] ; then
#     echo "lz complexity"
#     python ./scripts/lempel_ziv_complexity.py \
#         "--infile=$DATA_PATH/processed/prefilter.parquet" \
#         "--outfile=$DATA_PATH/processed/lz_complexity.parquet" \
#         "--first-frame=3" \
#         "--local" \
#         "--cores=$CORES" \
#         "--memory=$MEM_PER_CPU"
# fi

if [[ -n "$GIT_COMMIT" ]]; then
    echo $GIT_COMMIT > $DATA_PATH/run-environment/commit.hash;
else
    git rev-parse HEAD > $DATA_PATH/run-environment/commit.hash;
fi
