import sys
import os
import numpy as np
import pandas as pd
from finite_volume.advection import AdvectionSolver

# creating data directory if it doesn't exist
data_directory = "data/cases/"
path_to_data = data_directory + "disk.csv"
if os.path.exists(data_directory):
    if os.path.exists(path_to_data):
        raise BaseException(f"Existing data logged at {path_to_data}")
else:
    os.makedirs(data_directory)


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


def vortex(x, y):
    return -y, x


trials = 10
ns = [(32,), (64,), (128,)]
orders = [1, 2, 3, 4, 5, 6, 7, 8]
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
integrator_config_dict = {"ssprk3": 1, "rk3": 2, "rk4": 3}

list_of_data = []
solution_count = 0
for n in ns:
    for order in orders:
        for limiter_key, limiter_config in limiter_config_dict.items():
            courants = (
                [0.8, mpp_cfl[order]] if limiter_config["apriori_limiting"] else [0.8]
            )
            for courant in courants:
                for integrator_key, integrator_config in integrator_config_dict.items():
                    if integrator_config == 1:
                        flux_strategies = ["gauss-legendre", "transverse"]
                    elif limiter_key == "a priori":
                        flux_strategies = ["gauss-legendre"]
                    elif limiter_key == "a posteriori":
                        flux_strategies = ["transverse"]
                    for flux_strategy in flux_strategies:
                        times = []
                        for trial in range(trials):
                            enablePrint()
                            print(
                                f"n = {n}, order {order}",
                                f", flux strategy: {flux_strategy}",
                                f", limiting: {limiter_key}, courant = {courant}",
                                f", integrator: {integrator_key}",
                                f", trial {trial + 1}/{trials}",
                            )
                            blockPrint()
                            solver = AdvectionSolver(
                                u0="disk",
                                x=(-1, 1),
                                v=vortex,
                                T=2 * np.pi,
                                bc="dirichlet",
                                const=0,
                                log_every=100000,
                                n=n,
                                order=order,
                                courant=courant,
                                flux_strategy=flux_strategy,
                                apriori_limiting=limiter_config["apriori_limiting"],
                                aposteriori_limiting=limiter_config[
                                    "aposteriori_limiting"
                                ],
                                convex_aposteriori_limiting=limiter_config[
                                    "convex_aposteriori_limiting"
                                ],
                                load=False,
                            )
                            if integrator_config == 1:
                                solver.ssprk3()
                            elif integrator_config == 2:
                                solver.rk3()
                            elif integrator_config == 3:
                                solver.rk4()
                            else:
                                raise BaseException("invalid integrator")
                            times.append(solver.solution_time)
                        # gather data
                        times = np.asarray(times)
                        mean_time = np.mean(times)
                        std_time = np.std(times)
                        data = {}
                        data["n"] = n
                        data["order"] = order
                        data["flux strategy"] = flux_strategy
                        data["limiter"] = limiter_key
                        data["courant"] = courant
                        data["integrator"] = integrator_key
                        data["abs min"] = solver.abs_min
                        data["mean min"] = solver.mean_min
                        data["std min"] = solver.std_min
                        data["abs max"] = solver.abs_max
                        data["mean max"] = solver.mean_max
                        data["std max"] = solver.std_max
                        data["mean time"] = mean_time
                        data["std time"] = std_time
                        list_of_data.append(data)
                        # save data so far
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
