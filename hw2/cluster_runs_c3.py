"""Run on the cluster
NOTE: See local_config.template.py for a local_config template.
"""
import os
import glob

email = "msy290@nyu.edu"
directory="/scratch/msy290/hpml/hw2/"

run = 'C3'

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
		f.write("#SBATCH --mail-type=END,FAIL\n")
		f.write("#SBATCH --mail-user=%s\n" % email)
		f.write("module load anaconda3/5.3.1\n")
		f.write("source activate /home/msy290/.conda/envs/latest")
		f.write("#SBATCH --time=20:00:00")
		f.write("#SBATCH --mem=60GB")
		f.write("#SBATCH --nodes=1")
		f.write(jobcommand + "\n")

	s = "sbatch {}".format(os.path.join(slurm_scripts, jobname + ".slurm"))
	os.system(s)


job = {


		}


time = 48

nworkers = [0,4,8,12,16,20,24,28,32,36,40]

for w in nworkers:
	job['num-workers']=w
	jname = 'workers_{}'.format(w)
	train(job, jname)

