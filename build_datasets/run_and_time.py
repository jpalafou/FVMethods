import os
import numpy as np
import pandas as pd
import argparse
from finite_volume.advection import AdvectionSolver


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem")
    parser.add_argument("--limiter")
    parser.add_argument("--flux_strategy")
    parser.add_argument("--order")
    parser.add_argument("--n")
    parser.add_argument("--courant")
    parser.add_argument("--modify_time_step")
    parser.add_argument("--integrator")
    args = parser.parse_args()
    problem = args.problem
    n = int(args.n)
    order = int(args.order)
    courant = float(args.courant)
    limiter = args.limiter
    modify_time_step = args.modify_time_step == "True"
    flux_strategy = args.flux_strategy
    integrator = args.integrator

# limiter
if limiter == "a priori":
    apriori_limiting = True
    aposteriori_limiting = False
    convex_aposteriori_limiting = False
elif limiter == "classic a posteriori":
    apriori_limiting = False
    aposteriori_limiting = True
    convex_aposteriori_limiting = False
elif limiter == "convex a posteriori":
    apriori_limiting = False
    aposteriori_limiting = True
    convex_aposteriori_limiting = True

# problem setup
if problem == "composite":
    u0 = "composite"
    v = 1
    x = (0, 1)
    T = 1
    bc = "periodic"
    const = None
    smooth_extrema_detection = True
    flux_strategies = ["gauss-legendre"]
elif problem == "square2d":
    n = (n, n)
    u0 = "square"
    v = (2, 1)
    x = (0, 1)
    T = 1
    bc = "periodic"
    const = None
    smooth_extrema_detection = False
    flux_strategies = ["gauss-legendre", "transverse"]
elif problem == "disk":

    def vortex(x, y):
        return -y, x

    n = (n, n)
    u0 = "disk"
    v = vortex
    x = (-1, 1)
    T = 2 * np.pi
    bc = "dirichlet"
    const = 0
    smooth_extrema_detection = False
    flux_strategies = ["gauss-legendre", "transverse"]


# integer expression of grid size
nint = n if isinstance(n, int) else n[0]

# creating data directory if it doesn't exist
data_directory = f"data/cases/{problem}/{limiter}/{flux_strategy}/"
path_to_data = (
    data_directory
    + f"order{order}_n{nint}_courant{courant}_refine{modify_time_step}_{integrator}.csv"
)
try:
    os.makedirs(data_directory)
except OSError:
    pass

trials = 10
times = []
load = False
print(modify_time_step)
for trial in range(trials):
    load = True if trial == trials - 1 else load
    solver = AdvectionSolver(
        u0=u0,
        x=x,
        v=v,
        T=T,
        n=n,
        order=order,
        courant=courant,
        modify_time_step=modify_time_step,
        bc=bc,
        const=const,
        apriori_limiting=apriori_limiting,
        aposteriori_limiting=aposteriori_limiting,
        convex_aposteriori_limiting=convex_aposteriori_limiting,
        smooth_extrema_detection=smooth_extrema_detection,
        log_every=100000,
        load=load,
    )
    if integrator == "ssprk3":
        solver.ssprk3()
    elif integrator == "rk3":
        solver.rk3()
    elif integrator == "rk4":
        solver.rk4()
    else:
        raise BaseException("invalid integrator")
    times.append(solver.solution_time)
# compute time statistics
times = np.asarray(times)
mean_time = np.mean(times)
std_time = np.std(times)
# gather data
data = {}
data["n"] = nint
data["steps"] = solver.steps
data["order"] = order
data["flux strategy"] = flux_strategy
data["limiter"] = limiter
data["courant"] = courant
data["timestep"] = "adaptive" if modify_time_step else "fixed"
data["integrator"] = integrator
data["abs min"] = solver.abs_min
data["mean min"] = solver.mean_min
data["std min"] = solver.std_min
data["abs max"] = solver.abs_max
data["mean max"] = solver.mean_max
data["std max"] = solver.std_max
data["mean time"] = mean_time
data["std time"] = std_time
# save as csv
dataframe = pd.DataFrame([data])
dataframe.to_csv(path_to_data, index=False)
