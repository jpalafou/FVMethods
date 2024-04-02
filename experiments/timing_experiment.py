import argparse
from itertools import product
import os
import pandas as pd
from configs import limiting_schemes_2d, problem_configs, timing_solver_config
from finite_volume.advection import AdvectionSolver

limiting_schemes = limiting_schemes_2d

n_steps = 10
N = [32, 64, 128, 256, 512, 1024, 2048, 4096]

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", action="store_true")
args = parser.parse_args()
gpu = args.gpu

data = []

for n, p, lckey in product(N, range(8), ["ZS2D-M-Fdt", "FM2D-PP-T"]):
    solver = AdvectionSolver(
        n=(n,),
        order=p + 1,
        cupy=gpu,
        snapshot_dt=1.0,
        num_snapshots=1,
        **limiting_schemes[lckey],
        **problem_configs["sinus2d"],
        **timing_solver_config,
        load=True,
    )
    solver.snapshot_dt = 0
    for _ in range(n_steps):
        solver.snapshot_dt += solver.dt
    solver.rkorder(rk6=True if p > 3 else False)

    # save data
    assert solver.step_count == n_steps
    data_entry = dict(
        n=n,
        p=p,
        scheme=lckey,
        integrator=solver.integrator_name,
        n_steps=n_steps,
        solution_time=solver.solution_time,
        n_cells_updated_per_s=solver.n_cells * solver.step_count / solver.solution_time,
        device="gpu" if gpu else "cpu",
    )
    print(data_entry)
    data.append(data_entry)

# create dataframe
df = pd.DataFrame(data)

fname = "experiments/data/timing.csv"
if not os.path.isfile(fname):
    # save dataframe, if file does not exist write header
    df.to_csv(fname)
else:
    # it exists so append without writing the header
    df.to_csv(fname, mode="a", header=False)
