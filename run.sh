#!/bin/bash

set -a
source .env
set +a

usage()
{
  echo "usage: $0 -s # Run using singularity container"
  echo "       $0 -d # Run using docker container"
  echo "       $0 -b # Run using slurm batch scheduler"
  echo "       $0 -l # Run in local python environment"
}

while getopts "sdbl" flag; do
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

if [[ $DO_SINGULARITY = true  ]] ; then
  echo "Running using singularity"
  mkdir -p $SAVE_PATH
  singularity run --env-file .env \
                  --bind $DATA_PATH:/data \
                  --bind $SAVE_PATH:/results \
                  app.sif

elif [[ $DO_DOCKER = true  ]] ; then
  docker compose up --build

elif [[ $DO_SLURM = true  ]] ; then
  echo "Running using slurm batch scheduler"
  sbatch --cpus-per-task=$CORES \
         --mem-per-cpu="${MEM_PER_CPU}G" \
         ./slurm/pipeline.sh

elif [[ $DO_LOCAL = true  ]] ; then
  echo "Running in local anaconda environment"
  uv venv .venv
  source .venv.bin/activate
  uv sync --locked
  uv run main.py pipeline --root-dir=$DATA_PATH \
                          --save-dir=$SAVE_PATH \
                          --dataset=$DATASET \
                          --sample-rate=$SAMPLE_RATE \
                          --frame=$FRAME \
                          --hop=$HOP \
                          --n-fft=$N_FFT \
                          --cores=$CORES \
                          --memory=$MEM_PER_CPU \
                          --min-conf=$MIN_CONF \
                          --segment-duration=$SEGMENT_LEN \
                          --dc-correction=$DC_CORR \
                          --high-pass-filter=$HIGH_PASS_FILTER

else
  usage
fi
