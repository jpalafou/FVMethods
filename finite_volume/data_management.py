import os
import sys


def blockPrint():
    """
    function to enable printing
    """
    sys.stdout = open(os.devnull, "w")


def enablePrint():
    """
    function to disable printing
    """
    sys.stdout = sys.__stdout__


def build_job_script(job_number, time_limit, n, order, problem):
    bash_script = """#!/bin/bash
#SBATCH --job-name=build{}     # create a short name for your job
#SBATCH --nodes=1              # node count
#SBATCH --ntasks=1             # total number of tasks across all nodes
#SBATCH --cpus-per-task=1      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G       # memory per cpu-core (4G is default)
#SBATCH --time={}              # total run time limit (HH:MM:SS)
#SBATCH --mail-user=jpalafou@princeton.edu

module purge
module load anaconda3/2023.3
conda activate build

echo "running"
python build_datasets/build_dataset.py --n {} --order {} --problem {}
""".format(
        job_number, time_limit, n, order, problem
    )

    path_to_file = f"build_datasets/job_scripts/build{job_number}.slurm"
    with open(path_to_file, "w") as somefile:
        somefile.write(bash_script)
    return path_to_file
