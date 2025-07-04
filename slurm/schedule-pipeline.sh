#!/bin/bash 
#SBATCH -J soundade-pipeline
#SBATCH -o soundade-pipeline."%j".out
#SBATCH -e soundade-pipeline."%j".err
#Default in slurm
#SBATCH --mail-user $USER@sussex.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -p short

cd $HOME/SoundADE
bash $HOME/SoundADE/run-pipeline.sh -s -p $DATA_PATH
