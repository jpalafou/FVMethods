def build_job_script(
    name,
    time_limit,
    problem,
    limiter,
    flux_strategy,
    integrator,
):
    bash_script1 = """#!/bin/bash
#SBATCH --job-name={}     # create a short name for your job
#SBATCH --nodes=1              # node count
#SBATCH --ntasks=1             # total number of tasks across all nodes
#SBATCH --cpus-per-task=1      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G       # memory per cpu-core (4G is default)
#SBATCH --time={}              # total run time limit (HH:MM:SS)
#SBATCH --mail-user=jpalafou@princeton.edu

module purge
module load anaconda3/2023.3
conda activate build

echo "running"\n
""".format(
        name, time_limit
    )
    bash_script2 = (
        "python build_datasets/build_dataset.py "
        + "--problem {} ".format(problem)
        + "--limiter '{}' ".format(limiter)
        + "--flux_strategy {} ".format(flux_strategy)
        + "--integrator {} ".format(integrator)
    )

    path_to_file = f"build_datasets/job_scripts/{name}.slurm"
    with open(path_to_file, "w") as somefile:
        somefile.write(bash_script1 + bash_script2)
    return path_to_file
