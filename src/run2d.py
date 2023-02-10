import numpy as np
from util.initial_condition import initial_condition2D
from util.advection2d import AdvectionSolver2D

# inputs
ic_type = "sinus"  # initial condition type
a = [1, 1]  # tranpsort speed
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
x_interface = np.linspace(x_bounds[0], x_bounds[1], n)
x = 0.5 * (x_interface[:-1] + x_interface[1:])  # x at cell centers

# initial condition
u0 = initial_condition2D(x, ic_type)

for config in solution_configs:
    # time vector
    time_step_adjustment = ""
    rkorder = config["time order"]
    Dt = config["courant factor"] * h / max(a)
    if config["time order"] > 4:
        rkorder = 4
        time_step_adjustment = 10 ** (config["time order"] - 4)
        Dt = Dt / time_step_adjustment
        time_step_adjustment = f" + Dt / {time_step_adjustment}"
    n_time = int(np.ceil(T / Dt))
    t = np.linspace(0, T, num=n_time)
    # set up solution
    if config["solution scheme"] == "no limiter":
        solution = AdvectionSolver2D(
            u0=u0, t=t, h=h, a=a[0], b=a[1], order=config["spatial order"]
        )
    # time integration
    solution.rkn(rkorder)
