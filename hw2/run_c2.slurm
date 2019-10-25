#!/bin/bash

#SBATCH --job-name=jupyterTest2
#SBATCH --nodes=1
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=C2_cpu.txt

source /misc/kcgscratch1/ChoGroup/srikar/my_venv/bin/activate


python3 -B lab2.py 

