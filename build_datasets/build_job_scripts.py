import os
from finite_volume.data_management import build_job_script

path_to_executable = "build_datasets/run_jobs.sh"

try:
    os.remove(path_to_executable)
except OSError:
    pass

job_number = 1
for order in [1, 2, 3, 4, 5, 6, 7, 8]:
    for n in [128, 256, 512]:
        path_to_file = build_job_script(
            problem="composite",
            time_limit="01:00:00",
            n=n,
            order=order,
            job_number=job_number,
            )
        with open(path_to_executable, "a") as somefile:
            somefile.write("sbatch " + path_to_file + "\n")
        job_number += 1
    for n in [32, 64, 128]:
        path_to_file = build_job_script(
            problem="square2d",
            time_limit="03:00:00",
            n=n,
            order=order,
            job_number=job_number,
            )
        with open(path_to_executable, "a") as somefile:
            somefile.write("sbatch " + path_to_file + "\n")
        job_number += 1
    for n in [64, 128, 256]: 
        path_to_file = build_job_script(
            problem="disk",
            time_limit="10:00:00",
            n=n,
            order=order,
            job_number=job_number,
            )
        with open(path_to_executable, "a") as somefile:
            somefile.write("sbatch " + path_to_file + "\n")
        job_number += 1