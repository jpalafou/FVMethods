import numpy as np
import matplotlib.pyplot as plt
from util.advection1d import (
    AdvectionSolver,
    AdvectionSolver_nOrder_MPP,
    AdvectionSolver_nOrder_MPP_lite,
)

# configurations
global_config = {
    "mesh sizes": [64],
    "initial condition": "sinus",
    "courant factor": 0.5,
    "advection velocity": 1,
    "solution time": 2,
    "solution scheme": "no limiter",
}
solution_configs = [
    {
        "order": 1,
    },
    {
        "order": 2,
    },
    {
        "order": 3,
    },
    {
        "order": 4,
    },
    {
        "order": 5,
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
        if config["solution scheme"] == "no limiter":
            solution = AdvectionSolver(
                u0_preset=config["initial condition"],
                n=n,
                T=config["solution time"],
                a=config["advection velocity"],
                courant=config["courant factor"],
                order=config["order"],
            )
        elif config["solution scheme"] == "mpp":
            solution = AdvectionSolver_nOrder_MPP(
                u0_preset=config["initial condition"],
                n=n,
                T=config["solution time"],
                a=config["advection velocity"],
                courant=config["courant factor"],
                order=config["order"],
            )
        elif config["solution scheme"] == "mpp lite":
            solution = AdvectionSolver_nOrder_MPP_lite(
                u0_preset=config["initial condition"],
                n=n,
                T=config["solution time"],
                a=config["advection velocity"],
                courant=config["courant factor"],
                order=config["order"],
            )
        else:
            raise BaseException(
                f"invalid solution scheme {config['solution scheme']}"
            )
        # time integration
        solution.rkn(config["order"])
        # errors
        print(
            f"l1 error = {np.sum(np.abs(solution.u[-1] - solution.u[0])) / n}"
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
        label = (
            f"{config['solution scheme']} order {config['order']}"
            f" + {time_message}"
        )
        # plotting
        plt.plot(
            solution.x,
            solution.u[-1],
            "-",
            marker="o",
            mfc="none",
            label=label,
        )

# finish plot
plt.xlabel(r"$x$")
plt.ylabel(r"$\bar{u}$")
plt.legend()
plt.show()
