import numpy as np
import matplotlib.pyplot as plt
from util.initial_condition import initial_condition1d
from util.integrate import rk4_Dt_adjust
from util.advection1d import (
    AdvectionSolver,
    AdvectionSolver_nOrder_MPP,
    AdvectionSolver_nOrder_MPP_lite,
)


# inputs
ic_type = "sinus"  # initial condition type
a = 1  # tranpsort speed
x_bounds = [0, 1]  # spatial domain
T = 2  # solving time

# configurations
global_configs = {
    "mesh sizes": [16, 64],
    "error norm": "l1",
}
solution_configs = [
    {
        "spatial order": 4,
        "time order": 4,
        "courant factor": 0.5,
        "solution scheme": "no limiter",
    },
]

# calculate h's
h_list = [
    (x_bounds[1] - x_bounds[0]) / n for n in global_configs["mesh sizes"]
]
# begin plot with finest initial condition
x_interface = np.linspace(
    x_bounds[0], x_bounds[1], num=max(global_configs["mesh sizes"])
)
x = 0.5 * (x_interface[:-1] + x_interface[1:])  # x at cell centers
u0 = initial_condition1d(x, ic_type)
plt.plot(x, u0, label="initial condition")
for config in solution_configs:
    for n in global_configs["mesh sizes"]:
        # global config
        for key in config.keys():
            if key in global_configs.keys():
                config[key] = global_configs[key]
        # print update
        for key, item in config.items():
            print(f"{key}: {item}")
        print(f"mesh size: {n}")
        print()
        # array of x-values
        x_interface = np.linspace(x_bounds[0], x_bounds[1], num=n + 1)
        x = 0.5 * (x_interface[:-1] + x_interface[1:])  # x at cell centers
        h = (x_bounds[1] - x_bounds[0]) / n
        # time vector
        time_step_adjustment_label = ""
        rkorder = config["time order"]
        Dt = config["courant factor"] * h / np.abs(a)
        if config["time order"] > 4:
            rkorder = 4
            time_step_adjustment = rk4_Dt_adjust(
                h, x_bounds[1] - x_bounds[0], config["spatial order"]
            )
            Dt = Dt * time_step_adjustment
            time_step_adjustment_label = (
                r" + $\Delta t$ * " + f"{round(time_step_adjustment, 5)}"
            )
        n_time = int(np.ceil(T / Dt))
        t = np.linspace(0, T, num=n_time)
        # initial condition
        u0 = initial_condition1d(x, ic_type)
        # set up solution
        if config["solution scheme"] == "no limiter":
            solution = AdvectionSolver(
                u0=u0, t=t, h=h, a=a, order=config["spatial order"]
            )
        elif config["solution scheme"] == "mpp":
            solution = AdvectionSolver_nOrder_MPP(
                u0=u0, t=t, h=h, a=a, order=config["spatial order"]
            )
        elif config["solution scheme"] == "mpp lite":
            solution = AdvectionSolver_nOrder_MPP_lite(
                u0=u0, t=t, h=h, a=a, order=config["spatial order"]
            )
        else:
            raise BaseException(
                f"invalid solution scheme {config['solution scheme']}"
            )
        # time integration
        solution.rkn(rkorder)
        # label generation
        if rkorder == 1:
            time_message = "euler"
        else:
            time_message = f"rk{rkorder}{time_step_adjustment_label}"
        label = (
            f"{config['solution scheme']} order {config['spatial order']}"
            f" + {time_message}"
        )
        # plotting
        plt.plot(x, solution.u[-1], "-", marker="o", mfc="none", label=label)

# finish plot
plt.xlabel(r"$x$")
plt.ylabel(r"$\bar{u}$")
plt.legend()
plt.show()
