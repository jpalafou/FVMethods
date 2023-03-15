import numpy as np
import matplotlib.pyplot as plt
import warnings
from util.advection1d import AdvectionSolver

warnings.filterwarnings("ignore")

# configurations
global_config = {
    "mesh sizes": [32],
    "initial condition": "square",
    "courant factor": 0.5,
    "solution time": 1,
    "apriori limiting": None
}
solution_configs = [
    {
        "order": 5,
        "apriori limiting": "mpp",
    },
    {"order": 5, "apriori limiting": "mpp lite", "linetype": "o--"},
    {
        "order": 5,
        "apriori limiting": None,
    },
]

# initialize plot
solution = AdvectionSolver(
    u0_preset=global_config["initial condition"],
    n=max(global_config["mesh sizes"]),
)
plt.plot(solution.x, solution.u[0], "-", mfc="none", label="initial condition")
# run all cases
for config in solution_configs:
    for n in global_config["mesh sizes"]:
        # global config
        for key in global_config.keys():
            config[key] = global_config[key]
        # print update
        print("- - - - -")
        for key, item in config.items():
            print(f"{key}: {item}")
        print(f"mesh size: {n}")
        print()
        # define solution
        solution = AdvectionSolver(
            u0_preset=config["initial condition"],
            n=n,
            T=config["solution time"],
            a=config["advection velocity"],
            courant=config["courant factor"],
            order=config["order"],
            apriori=config["apriori limiting"],
        )
        # time integration
        solution.rkn(config["order"])
        # errors
        print(
            "l1 error = ",
            f"{round(np.sum(np.abs(solution.u[-1] - solution.u[0])) / n, 5)}",
        )
        print(
            f"final soluion min = {round(min(solution.u[-1]), 5)}, ",
            f"max = {round(max(solution.u[-1]), 5)}",
        )
        # label generation
        if config["order"] == 1:
            time_message = "euler"
        elif config["order"] > 1 and config["order"] <= 4:
            time_message = f"rk{config['order']}"
        else:
            time_message = (
                "rk4 + "
                + r"$\Delta t$"
                + f" * {round(solution.time_step_adjustment, 5)}"
            )
        limiter_message = solution.apriori if solution.apriori else "no"
        label = (
            f"{limiter_message} limiter order {config['order']}"
            f" + {time_message}"
        )
        # plotting
        linetype = "o-"
        if "linetype" in config.keys():
            linetype = config["linetype"]
        plt.plot(
            solution.x,
            solution.u[-1],
            linetype,
            mfc="none",
            label=label,
        )

# finish plot
plt.xlabel(r"$x$")
plt.ylabel(r"$\bar{u}$")
plt.legend()
plt.show()
