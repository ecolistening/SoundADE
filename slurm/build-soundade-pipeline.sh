#!/bin/bash 
#SBATCH -J build-soundade-pipeline
#SBATCH -o bsp."%j".out
#SBATCH -e bsp."%j".err
#Default in slurm
#SBATCH --mail-user $USER@sussex.ac.uk
#SBATCH --mail-type=ALL
#Request 30 minutes run time
#SBATCH -t 0:30:0
#SBATCH -p short
#SBATCH --mem 32G
#SBATCH --cpus-per-task 8

cd $HOME/SoundADE
apptainer build --ignore-fakeroot-command -F $HOME/SoundADE/pipeline.sif $HOME/SoundADE/pipeline.def
