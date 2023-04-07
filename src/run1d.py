import matplotlib.pyplot as plt
import warnings
from util.advection1d import AdvectionSolver

warnings.filterwarnings("ignore")

# configurations
n = 256
u0_preset = "composite"
T = 1
solutions = [
    AdvectionSolver(
        n=n,
        order=4,
        u0_preset=u0_preset,
        courant=0.166,
        T=T,
        apriori_limiting="mpp",
        smooth_extrema_detection=True,
    ),
    AdvectionSolver(
        n=n,
        order=4,
        u0_preset=u0_preset,
        courant=0.166,
        T=T,
        aposteriori_limiting=True,
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
for solution in solutions:
    # time integration
    solution.ssp_rk3()
    # errors
    print(f"l1 error = {solution.find_error('l1')}")
    print(
        f"final soluion min = {round(min(solution.u[-1]), 5)}, ",
        f"max = {round(max(solution.u[-1]), 5)}",
    )
    print("- - - - - - - - - -")
    # label generation
    if solution.order == 1:
        time_message = "euler"
    elif solution.order > 1 and solution.order < 4:
        time_message = f"rk{solution.order}"
    elif solution.order >= 4:
        time_message = "rk4"
    if solution.adujst_time_step and solution.order > 4:
        time_message += (
            " + "
            + r"$\Delta t$"
            + f" * {round(solution.time_step_adjustment, 5)}"
        )
    if solution.apriori_limiting:
        limiter_message = solution.apriori_limiting + " limiter"
    elif solution.aposteriori_limiting:
        limiter_message = "posteriori limiter"
    else:
        limiter_message = "no limiter"
    limiter_message += (
        " with smooth extrema detection"
        if solution.smooth_extrema_detection
        else ""
    )
    label = f"{limiter_message} order {solution.order}" f" + {time_message}"
    # plotting
    plt.plot(
        solution.x,
        solution.u[-1],
        "o--",
        mfc="none",
        label=label,
    )

# finish plot
plt.xlabel(r"$x$")
plt.ylabel(r"$\bar{u}$")
plt.legend()
plt.show()
