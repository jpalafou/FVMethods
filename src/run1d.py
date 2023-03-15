import numpy as np
import matplotlib.pyplot as plt
import warnings
from util.advection1d import AdvectionSolver

warnings.filterwarnings("ignore")

# configurations
order = 5
n = 32
u0_preset = "square"
solutions = [
    AdvectionSolver(n=n, order=order, u0_preset=u0_preset, apriori="mpp"),
    AdvectionSolver(
        n=n,
        order=order,
        u0_preset=u0_preset,
        apriori="mpp",
        smooth_extrema=True,
    ),
]

# initialize plot
plt.plot(
    solutions[0].x,
    solutions[0].u[0],
    "-",
    mfc="none",
    label="initial condition",
)
# run all cases
dashed = False
for solution in solutions:
    # time integration
    solution.rkorder()
    # errors
    print(
        "l1 error = ",
        f"{round(np.sum(np.abs(solution.u[-1] - solution.u[0])) / n, 5)}",
    )
    print(
        f"final soluion min = {round(min(solution.u[-1]), 5)}, ",
        f"max = {round(max(solution.u[-1]), 5)}",
    )
    print("- - - - - - - - - -")
    # label generation
    if solution.order == 1:
        time_message = "euler"
    elif solution.order > 1 and solution.order <= 4:
        time_message = f"rk{solution.order}"
    else:
        time_message = (
            "rk4 + "
            + r"$\Delta t$"
            + f" * {round(solution.time_step_adjustment, 5)}"
        )
    limiter_message = (
        solution.apriori if solution.apriori else "no"
    ) + " limiter"
    limiter_message += (
        " with smooth extrema detection" if solution.smooth_extrema else ""
    )
    label = f"{limiter_message} order {solution.order}" f" + {time_message}"
    # plotting
    if dashed:
        linetype = "o--"
    else:
        linetype = "o-"
    dashed = not dashed
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
