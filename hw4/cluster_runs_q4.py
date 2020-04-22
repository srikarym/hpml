"""Run on the cluster
NOTE: See local_config.template.py for a local_config template.
"""
import os
import glob

email = "msy290@nyu.edu"
directory="/misc/kcgscratch1/ChoGroup/srikar/hpml/hw4"

run = 'Q5'

slurm_logs = os.path.join(directory, "slurm_logs",run)
slurm_scripts = os.path.join(directory, "slurm_scripts",run)
log_dir = os.path.join(directory, 'logs',run)

if not os.path.exists(slurm_logs):
	os.makedirs(slurm_logs)
if not os.path.exists(slurm_scripts):
	os.makedirs(slurm_scripts)
if not os.path.exists(log_dir):
	os.makedirs(log_dir)


def train(time=24):

	job = {'epochs':5}
	jobname = 'Q4_allruns'

	slurmfile = os.path.join(slurm_scripts, jobname + '.slurm')
	with open(slurmfile, 'w') as f:
		f.write("#!/bin/bash\n")
		f.write("#SBATCH --job-name" + "=" +'run4' + "\n")
		f.write("#SBATCH --output=%s\n" % os.path.join(slurm_logs, jobname + ".out"))
		f.write("#SBATCH --gres=gpu:4\n")
		f.write('#SBATCH --constraint=gpu_12gb\n')
		f.write("#SBATCH --time=2:00:00\n")
		f.write("#SBATCH --mem=30GB\n")
		f.write("#SBATCH --nodes=1\n")
		f.write("#SBATCH --cpus-per-task=4\n")
		f.write("source /misc/kcgscratch1/ChoGroup/srikar/my_venv/bin/activate\n")
		f.write('module load cuda-10.1\n')

		for bs,n_gpus in [[128,1],[512,4]]:

			gpu_ids = ','.join(str(a) for a in range(n_gpus))
			
			flags = {k:v for k,v in job.items()}
			flags['batch-size'] = bs*n_gpus
			flags['gpu-id'] = gpu_ids

			jobcommand = "python3 -B lab4.py "
			args = ["--%s %s" % (flag, str(flags[flag])) 
					for flag in sorted(flags.keys())]

			jobcommand += " ".join(args) + '|& tee ' + \
				os.path.join(log_dir, f'bs_{bs}_{n_gpus}.txt')
			f.write(jobcommand + "\n")

	s = "sbatch {}".format(os.path.join(slurm_scripts, jobname + ".slurm"))
	os.system(s)


train()

