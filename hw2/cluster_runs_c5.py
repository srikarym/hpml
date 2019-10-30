"""Run on the cluster
NOTE: See local_config.template.py for a local_config template.
"""
import os
import glob

email = "msy290@nyu.edu"
directory="/misc/kcgscratch1/ChoGroup/srikar/hpml/hw2"

run = 'C5'

slurm_logs = os.path.join(directory, "slurm_logs",run)
slurm_scripts = os.path.join(directory, "slurm_scripts",run)

# logdir = os.path.join(directory, "logs",run)

if not os.path.exists(slurm_logs):
	os.makedirs(slurm_logs)
if not os.path.exists(slurm_scripts):
	os.makedirs(slurm_scripts)


def train(flags, jobname=None, time=24):

	jobcommand = "python3 -B lab2.py "
	args = ["--%s %s" % (flag, str(flags[flag])) for flag in sorted(flags.keys())]
	jobcommand += " ".join(args)

	slurmfile = os.path.join(slurm_scripts, jobname + '.slurm')
	with open(slurmfile, 'w') as f:
		f.write("#!/bin/bash\n")
		f.write("#SBATCH --job-name" + "=" +jobname + "\n")
		f.write("#SBATCH --output=%s\n" % os.path.join(slurm_logs, jobname + ".out"))
		f.write("#SBATCH --gres=gpu:1\n")
		f.write("source /misc/kcgscratch1/ChoGroup/srikar/my_venv/bin/activate\n")
		f.write("#SBATCH --time=20:00:00\n")
		f.write("#SBATCH --mem=60GB\n")
		f.write("#SBATCH --nodes=1\n")
		f.write(jobcommand + "\n")

	s = "sbatch {}".format(os.path.join(slurm_scripts, jobname + ".slurm"))
	os.system(s)


job = {'num-workers':12}


time = 48

cuda = [1]

for c in cuda:
	job['cuda']=c
	jname = 'cuda_{}'.format(c)
	train(job, jname)

