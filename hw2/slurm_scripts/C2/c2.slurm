#!/bin/bash
#SBATCH --job-name=c2
#SBATCH --output=/misc/kcgscratch1/ChoGroup/srikar/hpml/hw2/slurm_logs/C2/c2.out
#SBATCH --gres=gpu:1
source /misc/kcgscratch1/ChoGroup/srikar/my_venv/bin/activate
#SBATCH --time=20:00:00
#SBATCH --mem=60GB
#SBATCH --nodes=1
python3 -B lab2.py --cuda 1
