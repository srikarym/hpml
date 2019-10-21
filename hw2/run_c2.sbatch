#!/bin/bash

#SBATCH --job-name=jupyterTest2
#SBATCH --nodes=1
#SBATCH --mem=60GB

#SBATCH --time=12:00:00
#SBATCH --output=slurm_output.out
module purge

module load anaconda3/5.3.1

source activate /home/msy290/.conda/envs/latest


python3 lab2.py

