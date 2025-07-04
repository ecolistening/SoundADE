#!/bin/bash 
#SBATCH -J build-soundade-environment
#SBATCH -o bss."%j".out
#SBATCH -e bss."%j".err
#Default in slurm
#SBATCH --mail-user $USER@sussex.ac.uk
#SBATCH --mail-type=ALL
#Request 30 minutes run time
#SBATCH -t 0:30:0
#SBATCH -p short
#SBATCH --mem 64G
#SBATCH --cpus-per-task 8

cd $HOME/SoundADE
apptainer build --ignore-fakeroot-command -F $HOME/SoundADE/soundade-environment.sif $HOME/SoundADE/soundade-environment.def
