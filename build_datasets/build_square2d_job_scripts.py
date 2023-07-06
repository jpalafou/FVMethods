import os
import numpy as np
from finite_volume.data_management import build_job_script

path_to_executable = "build_datasets/run_square2d_jobs.sh"

try:
    os.remove(path_to_executable)
except OSError:
    pass

job_number = 1
mpp_cfl = {1: 0.5, 2: 0.5, 3: 0.166, 4: 0.166, 5: 0.0833, 6: 0.0833, 7: 0.05, 8: 0.05}
for limiter in ["a priori", "classic a posteriori", "convex a posteriori"]:
    for flux_strategy in ["gauss-legendre", "transverse"]:
        for order in [1, 2, 3, 4, 5, 6, 7, 8]:
            for n in [32, 64, 128]:
                courants = [0.8, mpp_cfl[order]] if limiter == "a priori" else [0.8]
                for courant in courants:
                    for integrator in ["ssprk3", "rk3", "rk4"]:
                        path_to_file = build_job_script(
                            name=f"square2d-{job_number}",
                            time_limit="{:02d}:00:00".format(
                                int(np.ceil(n**2 / 6500))
                            ),
                            problem="square2d",
                            limiter=limiter,
                            flux_strategy=flux_strategy,
                            order=order,
                            n=n,
                            courant=courant,
                            modify_time_step=False,
                            integrator=integrator,
                        )
                        with open(path_to_executable, "a") as somefile:
                            somefile.write("sbatch " + path_to_file + "\n")
                        job_number += 1
