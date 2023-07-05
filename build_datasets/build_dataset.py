import os
import numpy as np
import pandas as pd
import argparse
from finite_volume.advection import AdvectionSolver
from finite_volume.utils import blockPrint
from finite_volume.utils import enablePrint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem')
    parser.add_argument('--n')
    parser.add_argument('--order')
    args = parser.parse_args()
    problem = args.problem
    n = int(args.n)
    order = int(args.order)


if problem == 'composite':
    u0 = "composite"
    v = 1
    x = (0, 1)
    T = 1
    bc = 'periodic'
    const = None
    smooth_extrema_detection = True
    flux_strategies = ['gauss-legendre']
elif problem == 'square2d':
    n = (n, n)
    u0 = "square"
    v = 1
    x = (0, 1)
    T = 1
    bc = 'periodic'
    const = None
    smooth_extrema_detection = False
    flux_strategies = ['gauss-legendre', 'transverse']
elif problem == 'disk':
    def vortex(x, y):
        return -y, x
    n = (n, n)
    u0 = "disk"
    v = vortex
    x = (-1, 1)
    T = 2 * np.pi
    bc = 'dirichlet'
    const = 0
    smooth_extrema_detection = False
    flux_strategies = ['gauss-legendre', 'transverse']

# creating data directory if it doesn't exist
data_directory = f"data/cases/{problem}/"
path_to_data = data_directory + f"order{order}_n{n}.csv"
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

trials = 10
mpp_cfl = {1: 0.5, 2: 0.5, 3: 0.166, 4: 0.166, 5: 0.0833, 6: 0.0833, 7: 0.05, 8: 0.05}
limiter_config_dict = {
    "a priori": {
        "apriori_limiting": True,
        "aposteriori_limiting": False,
        "convex_aposteriori_limiting": False,
    },
    "classic a posteriori": {
        "apriori_limiting": False,
        "aposteriori_limiting": False,
        "convex_aposteriori_limiting": True,
    },
    "convex a posteriori": {
        "apriori_limiting": False,
        "aposteriori_limiting": True,
        "convex_aposteriori_limiting": True,
    },
}
integrator_configs = ['ssprk3', 'rk3', 'rk4']

list_of_data = []
solution_count = 0
for limiter_key, limiter_config in limiter_config_dict.items():
    courants = (
        [0.8, mpp_cfl[order]] if limiter_config["apriori_limiting"] else [0.8]
    )
    for courant in courants:
        for integrator_config in integrator_configs:
            for flux_strategy in flux_strategies:
                load = False
                times = []
                for trial in range(trials):
                    enablePrint()
                    print(
                        f"n = {n}, order {order}, limiting: {limiter_key}",
                        f", courant = {courant}, integrator: {integrator_config}",
                        f", trial {trial + 1}/{trials}",
                    )
                    blockPrint()
                    load = True if trial == trials - 1 else load
                    solver = AdvectionSolver(
                        u0=u0,
                        x=x,
                        v=v,
                        T=T,
                        n=n,
                        order=order,
                        courant=courant,
                        bc = bc,
                        const = const,
                        apriori_limiting=limiter_config["apriori_limiting"],
                        aposteriori_limiting=limiter_config["aposteriori_limiting"],
                        convex_aposteriori_limiting=limiter_config[
                            "convex_aposteriori_limiting"
                        ],
                        smooth_extrema_detection=smooth_extrema_detection,
                        log_every=100000,
                        load=load,
                    )
                    if integrator_config == 'ssprk3':
                        solver.ssprk3()
                    elif integrator_config == 'rk3':
                        solver.rk3()
                    elif integrator_config == 'rk4':
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
                data["n"] = n
                data["steps"] = solver.steps
                data["order"] = order
                data["flux strategy"] = flux_strategy
                data["limiter"] = limiter_key
                data["courant"] = courant
                data["integrator"] = integrator_config
                data["abs min"] = solver.abs_min
                data["mean min"] = solver.mean_min
                data["std min"] = solver.std_min
                data["abs max"] = solver.abs_max
                data["mean max"] = solver.mean_max
                data["std max"] = solver.std_max
                data["mean time"] = mean_time
                data["std time"] = std_time
                list_of_data.append(data)
                # save every 10 solutions
                solution_count += 1
                if solution_count % 10 == 0:
                    enablePrint()
                    print(f"Saving {solution_count} solution results")
                    blockPrint()
                    dataframe = pd.DataFrame(list_of_data)
                    dataframe.to_csv(path_to_data, index=False)
# save as csv
dataframe = pd.DataFrame(list_of_data)
dataframe.to_csv(path_to_data, index=False)
