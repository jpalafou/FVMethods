import os
import numpy as np
from finite_volume.data_management import build_job_script

path_to_executable = "build_datasets/run_disk_jobs.sh"

try:
    os.remove(path_to_executable)
except OSError:
    pass

job_number = 1
mpp_cfl = {1: 0.5, 2: 0.5, 3: 0.166, 4: 0.166, 5: 0.0833, 6: 0.0833, 7: 0.05, 8: 0.05}
for limiter in ["a priori", "classic a posteriori", "convex a posteriori"]:
    for flux_strategy in ["gauss-legendre", "transverse"]:
        for order in [1, 2, 3, 4, 5, 6, 7, 8]:
            for n in [64, 128, 256]:
                modify_time_step_configs = (
                    [True, False] if limiter == "a priori" else [False]
                )
                for modify_time_step in modify_time_step_configs:
                    for integrator in ["ssprk3", "rk3", "rk4"]:
                        path_to_file = build_job_script(
                            name=f"disk-{job_number}",
                            time_limit="{:02d}:00:00".format(
                                int(np.ceil(n**2 / 6500))
                            ),
                            problem="disk",
                            limiter=limiter,
                            flux_strategy=flux_strategy,
                            order=order,
                            n=n,
                            courant=0.8,
                            modify_time_step=modify_time_step,
                            integrator=integrator,
                        )
                        with open(path_to_executable, "a") as somefile:
                            somefile.write("sbatch " + path_to_file + "\n")
                        job_number += 1
