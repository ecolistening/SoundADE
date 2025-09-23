#!/bin/bash

set -a
source .env
set +a
mkdir -p ./logs

sbatch --cpus-per-task=$CORES --mem-per-cpu=$MEM_PER_CPU ./slurm/jobs/pipeline.job
