import numpy as np
import matplotlib.pyplot as plt
from util.initial_condition import initial_condition1D
from util.advection import (
    AdvectionSolver,
    AdvectionSolver_nOrder_MPP,
    AdvectionSolver_nOrder_MPP_lite,
)


# inputs
ic_type = "sinus"  # initial condition type
a = 1  # tranpsort speed
n = 33  # number of cells
x_bounds = [0, 1]  # spatial domain
T = 2  # solving time

# configurations
solution_configs = [
    {
        "spatial order": 4,
        "time order": 4,
        "courant factor": 0.5,
        "solution scheme": "no limiter",
    },
]

# array of x-values
h = (x_bounds[1] - x_bounds[0]) / n
x_interface = np.arange(x_bounds[0], x_bounds[1] + h, h)
x = 0.5 * (x_interface[:-1] + x_interface[1:])  # x at cell centers

# initial condition
u0 = initial_condition1D(x, ic_type)

# begin plot
plt.plot(x, u0, label="initial condition")

for config in solution_configs:
    # time vector
    time_step_adjustment = ""
    rkorder = config["time order"]
    Dt = config["courant factor"] * h / a
    if config["time order"] > 4:
        rkorder = 4
        time_step_adjustment = 10 ** (config["time order"] - 4)
        Dt = Dt / time_step_adjustment
        time_step_adjustment = f" + Dt / {time_step_adjustment}"
    n_time = int(np.ceil(T / Dt))
    t = np.linspace(0, T, num=n_time)
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
    # plot
    plt.plot(
        x,
        solution.u[-1],
        "-",
        marker="o",
        mfc="none",
        label=(
            f"{config['solution scheme']} order "
            f"{config['spatial order']} + "
            f"rk{config['time order']}"
            f"{time_step_adjustment}"
        ),
    )

# finish plot
plt.xlabel(r"$x$")
plt.ylabel(r"$\bar{u}$")
plt.legend()
plt.show()
