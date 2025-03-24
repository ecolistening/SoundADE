#!/bin/bash 
#SBATCH -J soundade-pipeline
#SBATCH -o soundade-pipeline."%j".out
#SBATCH -e soundade-pipeline."%j".err
#Default in slurm
#SBATCH --mail-user $USER@sussex.ac.uk
#SBATCH --mail-type=ALL
#Request 4 minutes run time
#SBATCH -t 0:4:0
#SBATCH -p short

cd $HOME/SoundADE
bash $HOME/SoundADE/run-pipeline.sh -s
