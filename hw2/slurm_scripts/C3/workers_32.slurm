#!/bin/bash
#SBATCH --job-name=workers_32
#SBATCH --output=/misc/kcgscratch1/ChoGroup/srikar/hpml/hw2/slurm_logs/C3/workers_32.out
#SBATCH --gres=gpu:1
source /misc/kcgscratch1/ChoGroup/srikar/my_venv/bin/activate
#SBATCH --time=20:00:00
#SBATCH --mem=60GB
#SBATCH --nodes=1
python3 -B lab2.py --cuda 1 --num-workers 32
