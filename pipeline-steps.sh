#!/bin/bash
CONDA_PATH=/opt/conda

IFS=',' read -ra STEPS_ARRAY <<< "$STEPS"

# Option to use local conda environments
while getopts "l" flag; do
    case ${flag} in
        l) CONDA_PATH=$HOME/.conda
           ;;
    esac
done

# Sets and saves details of run environment for an individual step of the pipeline
#
# Takes a single label argument and labels the logs and folders with that label
# $1 - label/name of the current pipeline step
#
# Example
#
#   set_run_environment "process_files"
#
# Effects:
# STEP_LOG will be set with the path to the log file "process_files.log"
# PROFILE_PATH will be set to local copy and sourced
set_run_environment () {
    mkdir -p $DATA_PATH/processed/run-environment/"$1"
    PROFILE_PATH=$DATA_PATH/processed/run-environment/"$1"/profile.env
    cp $DATA_PATH/processed/run-environment/profile.env $PROFILE_PATH
    source $PROFILE_PATH
    cp $DATA_PATH/processed/run-environment/settings.env $DATA_PATH/processed/run-environment/"$1"/settings.env
    #Capture environment and git hash for reproducibility
    conda run -n soundade conda env export -c conda-forge -c anaconda -c defaults > $DATA_PATH/processed/run-environment/"$1"/environment.yml
    if [[ -n "$GIT_COMMIT" ]]; then
        echo $GIT_COMMIT > $DATA_PATH/processed/run-environment/"$1"/commit.hash;
    else
        git rev-parse HEAD > $DATA_PATH/processed/run-environment/"$1"/commit.hash;
    fi
    STEP_LOG=$DATA_PATH/processed/run-environment/"$1".log
    date > $DATA_PATH/processed/run-environment/"$1"/step-timestamp.txt
}

cd $CODE_PATH
for STEP in "${STEPS_ARRAY[@]}"; do
    if [[ $STEP -eq 0 ]] ; then
        echo -e "\nProcess audio files\n------"
        set_run_environment "process_files"
        $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/process_files.py \
                                             "--indir=$DATA_PATH" \
                                             "--outfile=$DATA_PATH/processed/processed.parquet" \
                                             "--dataset=$DATASET" \
                                             "--frame=$FRAME" \
                                             "--hop=$HOP" \
                                             "--n_fft=$N_FFT" \
                                             "--local" \
                                             "--cores=$CORES" \
                                             "--memory=$MEM_PER_CPU" \
                                             2>&1 | tee $STEP_LOG
    fi

    if [[ $STEP -eq 1 ]] ; then
        echo -e "\nAppend solar metadata\n------"
        set_run_environment "append_metadata"
        $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/append_metadata.py \
                                            "--infile=$DATA_PATH/processed/processed.parquet" \
                                            "--outfile=$DATA_PATH/processed/solar.parquet" \
                                            "--sitesfile=$DATA_PATH/site_locations.parquet" \
                                            "--local" \
                                            -F -C -T \
                                            "--cores=$CORES" \
                                            "--memory=$MEM_PER_CPU" \
                                            2>&1 | tee $STEP_LOG
    fi

    if [[ $STEP -eq 2 ]] ; then
        echo -e "\nPrefilter\n-------"
        set_run_environment "prefilter"
        $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/prefilter.py \
                                            "--infile=$DATA_PATH/processed/solar.parquet" \
                                            "--outfile=$DATA_PATH/processed/prefilter.parquet" \
                                            "--local" \
                                            "--cores=$CORES" \
                                            "--memory=$MEM_PER_CPU" \
                                            2>&1 | tee $STEP_LOG
    fi

    if [[ $STEP -eq 3 ]] ; then
        echo -e "\nConvert to long\n------"
        set_run_environment "to_long"
        $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/to_long.py \
                                            "--infile=$DATA_PATH/processed/prefilter.parquet" \
                                            "--outfile=$DATA_PATH/processed/to_long.parquet" \
                                            "--first-frame=3" \
                                            "--local" \
                                            "--cores=$CORES" \
                                            "--memory=$MEM_PER_CPU" \
                                            2>&1 | tee $STEP_LOG
    fi

    if [[ $STEP -eq 4 ]] ; then
        echo -e "\nSummary stats\n------"
        set_run_environment "summary_stats"
        $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/summary_stats.py \
                                            "--infile=$DATA_PATH/processed/to_long.parquet" \
                                            "--outfile=$DATA_PATH/processed/summary_stats.parquet" \
                                            "--long" \
                                            "--local" \
                                            "--cores=$CORES" \
                                            "--memory=$MEM_PER_CPU" \
                                            2>&1 | tee $STEP_LOG
    fi

    if [[ $STEP -eq 5 ]] ; then
        echo -e "\nHistograms\n------"
        set_run_environment "histograms"
        $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/astrograms.py \
                                            "--infile=$DATA_PATH/processed/to_long.parquet" \
                                            "--outfile=$DATA_PATH/processed/histograms.parquet" \
                                            "--histogram" \
                                            "--long" \
                                            "--local" \
                                            "--cores=$CORES" \
                                            "--memory=$MEM_PER_CPU" \
                                            2>&1 | tee $STEP_LOG
    fi

    if [[ $STEP -eq 6 ]] ; then
        echo -e "\nAstrograms\n------"
        set_run_environment "astrograms"
        $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/astrograms.py \
                                            "--infile=$DATA_PATH/processed/to_long.parquet" \
                                            "--outfile=$DATA_PATH/processed/astrograms.parquet" \
                                            "--astrogram" \
                                            "--long" \
                                            "--local" \
                                            "--cores=$CORES" \
                                            "--memory=$MEM_PER_CPU" \
                                            2>&1 | tee $STEP_LOG
    fi

    if [[ $STEP -eq 7 ]] ; then
        echo -e "\nLZ complexity\n-------"
        set_run_environment "lz_complexity"
        $CONDA_PATH/envs/soundade/bin/python $CODE_PATH/scripts/lempel_ziv_complexity.py \
                                            "--infile=$DATA_PATH/processed/prefilter.parquet" \
                                            "--outfile=$DATA_PATH/processed/lz_complexity.parquet" \
                                            "--first-frame=3" \
                                            "--local" \
                                            "--cores=$CORES" \
                                            "--memory=$MEM_PER_CPU" \
                                            2>&1 | tee $STEP_LOG
    fi
done

# Remove unneeded copies of run-environment
rm $DATA_PATH/processed/run-environment/profile.env
rm $DATA_PATH/processed/run-environment/settings.env
