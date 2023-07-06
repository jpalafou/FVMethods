import os
from finite_volume.data_management import build_job_script

path_to_executable = "build_datasets/list_of_jobs.sh"

try:
    os.remove(path_to_executable)
except OSError:
    pass

limiters = ["a priori", "classic a posteriori", "convex a posteriori"]
integrators = ["ssprk3", "rk3", "rk4"]
flux_strategies = ["gauss-legendre", "transverse"]

# COMPOSITE
job_number = 1
for limiter in limiters:
    for integrator in integrators:
        path_to_file = build_job_script(
            name=f"composite-{job_number}",
            time_limit="01:00:00",
            problem="composite",
            limiter=limiter,
            flux_strategy="gauss-legendre",
            integrator=integrator,
        )
        with open(path_to_executable, "a") as somefile:
            somefile.write("sbatch " + path_to_file + "\n")
        job_number += 1


# SQUARE 2D
job_number = 1
for limiter in limiters:
    for integrator in integrators:
        for flux_strategy in flux_strategies:
            path_to_file = build_job_script(
                name=f"square2d-{job_number}",
                time_limit="04:00:00",
                problem="square2d",
                limiter=limiter,
                flux_strategy=flux_strategy,
                integrator=integrator,
            )
            with open(path_to_executable, "a") as somefile:
                somefile.write("sbatch " + path_to_file + "\n")
            job_number += 1

# DISK
job_number = 1
for limiter in limiters:
    for integrator in integrators:
        for flux_strategy in flux_strategies:
            path_to_file = build_job_script(
                name=f"disk-{job_number}",
                time_limit="24:00:00",
                problem="disk",
                limiter=limiter,
                flux_strategy=flux_strategy,
                integrator=integrator,
            )
            with open(path_to_executable, "a") as somefile:
                somefile.write("sbatch " + path_to_file + "\n")
            job_number += 1
