#!/bin/bash
#SBATCH --job-name=workers_0
#SBATCH --output=/scratch/msy290/hpml/hw2/slurm_logs/C3/workers_0.out
#SBATCH --time=20:00:00
#SBATCH --mem=60GB
#SBATCH --nodes=1
module load anaconda3/5.3.1
source activate /home/msy290/.conda/envs/latest
python3 -B lab2.py --num-workers 0
