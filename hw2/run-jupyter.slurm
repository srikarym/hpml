#!/bin/bash

#SBATCH --qos=batch
#SBATCH --job-name=jupyterTest2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_output.out

module load cuda-10.0
module load gcc-9.1
source /misc/kcgscratch1/ChoGroup/srikar/my_venv/bin/activate

port=$(shuf -i 10000-12000 -n 1)



cat<<EOF


ssh -F ~/ssh_config -L $port:localhost:$port $(hostname)




EOF


jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)




