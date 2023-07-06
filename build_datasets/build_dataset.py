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
    parser.add_argument("--integrator")
    args = parser.parse_args()
    problem = args.problem
    limiter = args.limiter
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
    ns = [128, 256, 512]
    ndim = 1
elif problem == "square2d":
    u0 = "square"
    v = (2, 1)
    x = (0, 1)
    T = 1
    bc = "periodic"
    const = None
    smooth_extrema_detection = False
    flux_strategies = ["gauss-legendre", "transverse"]
    ns = [32, 64, 128]
    ndim = 2
elif problem == "disk":

    def vortex(x, y):
        return -y, x

    u0 = "disk"
    v = vortex
    x = (-1, 1)
    T = 2 * np.pi
    bc = "dirichlet"
    const = 0
    smooth_extrema_detection = False
    flux_strategies = ["gauss-legendre", "transverse"]
    ns = [64, 128, 256]
    ndim = 2

# orders
orders = [1, 2, 3, 4, 5, 6, 7, 8]
mpp_cfl = {1: 0.5, 2: 0.5, 3: 0.166, 4: 0.166, 5: 0.0833, 6: 0.0833, 7: 0.05, 8: 0.05}

# creating data directory if it doesn't exist
data_directory = f"data/cases/{problem}/"
path_to_data = data_directory + f"{limiter}_{flux_strategy}_{integrator}.csv"
try:
    os.makedirs(data_directory)
except OSError:
    pass

steps_per_trial = 20
trials = 10
list_of_data = []
for order in orders:
    courants = [0.8, mpp_cfl[order]] if limiter == "a priori" else [0.8]
    for courant in courants:
        for n in ns:
            if ndim == 1:
                ntuple = n
            elif ndim == 2:
                ntuple = (n, n)
            # print configurations
            print(f"n = {n}, order {order}, courant = {courant}")
            # time trials
            print("\ttime trial")
            times = []
            for _ in range(trials):
                # initialize solver for time trials
                solver = AdvectionSolver(
                    u0=u0,
                    x=x,
                    v=v,
                    T=T,
                    n=ntuple,
                    order=order,
                    courant=courant,
                    bc=bc,
                    const=const,
                    apriori_limiting=apriori_limiting,
                    aposteriori_limiting=aposteriori_limiting,
                    convex_aposteriori_limiting=convex_aposteriori_limiting,
                    smooth_extrema_detection=smooth_extrema_detection,
                    log_every=100000,
                    load=False,
                )
                # hack solving time to reduce number of time steps
                reduced_T = 0
                for _ in range(steps_per_trial):
                    reduced_T += solver.dt
                solver.T = reduced_T
                # integrate
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
            times_per_cell = times / solver.n_cells / solver.steps
            mean_tpc = np.mean(times_per_cell)
            std_tpc = np.std(times_per_cell)
            # complete solution
            print("\tcomplete solution")
            # initialize new solver for complete solution
            solver = AdvectionSolver(
                u0=u0,
                x=x,
                v=v,
                T=T,
                n=ntuple,
                order=order,
                courant=courant,
                bc=bc,
                const=const,
                apriori_limiting=apriori_limiting,
                aposteriori_limiting=aposteriori_limiting,
                convex_aposteriori_limiting=convex_aposteriori_limiting,
                smooth_extrema_detection=smooth_extrema_detection,
                log_every=100000,
                load=True,
            )
            # integrate
            if integrator == "ssprk3":
                solver.ssprk3()
            elif integrator == "rk3":
                solver.rk3()
            elif integrator == "rk4":
                solver.rk4()
            else:
                raise BaseException("invalid integrator")
            # gather data
            data = {}
            data["n"] = n
            data["steps"] = solver.steps
            data["order"] = order
            data["flux strategy"] = flux_strategy
            data["limiter"] = limiter
            data["courant"] = courant
            data["integrator"] = integrator
            data["abs min"] = solver.abs_min
            data["mean min"] = solver.mean_min
            data["std min"] = solver.std_min
            data["abs max"] = solver.abs_max
            data["mean max"] = solver.mean_max
            data["std max"] = solver.std_max
            data["mean time/cell (s)"] = mean_tpc
            data["std time/cell (s)"] = std_tpc
            list_of_data.append(data)
            # save as csv
            dataframe = pd.DataFrame(list_of_data)
            dataframe.to_csv(path_to_data, index=False)
