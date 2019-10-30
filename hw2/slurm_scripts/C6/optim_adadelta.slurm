#!/bin/bash
#SBATCH --job-name=optim_adadelta
#SBATCH --output=/misc/kcgscratch1/ChoGroup/srikar/hpml/hw2/slurm_logs/C6/optim_adadelta.out
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_12gb&turing"
source /misc/kcgscratch1/ChoGroup/srikar/my_venv/bin/activate
#SBATCH --time=20:00:00
#SBATCH --mem=60GB
#SBATCH --nodes=1
python3 -B lab2.py --cuda 1 --num-workers 12 --optimizer adadelta
