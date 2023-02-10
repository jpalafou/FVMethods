import numpy as np
import matplotlib.pyplot as plt
from initial_condition import initial_condition
from util.advection import AdvectionSolver_nOrder_MPP


# input
ic_type = "square"  # initial condition type
a = 1  # tranpsort speed
x_bounds = [0, 1]  # spatial domain
h = 0.02  # grid size
T = 10  # solving time


# array of t-values
def time_vector(C, a, h):
    Dt = C * h / a  # time step size
    return np.arange(0, T + Dt, Dt)


# array of x-values
x_interface = np.arange(x_bounds[0], x_bounds[1] + h, h)
x = 0.5 * (x_interface[:-1] + x_interface[1:])  # x at cell centers

# initial values of u
u0 = initial_condition(x, ic_type)

# solution
order = 8
Cs = [1.1, 0.05, 0.01]
solutions = [
    AdvectionSolver_nOrder_MPP(
        u0=u0, t=time_vector(C, a, h), h=h, a=a, order=order
    )
    for C in Cs
]
solutions_rk4 = [
    AdvectionSolver_nOrder_MPP(
        u0=u0, t=time_vector(C, a, h), h=h, a=a, order=order
    )
    for C in Cs
]

# plot
plt.plot(x, u0, label="initial condition")
for i in range(len(Cs)):
    solutions[i].ssp_rk3()
    plt.plot(
        x,
        solutions[i].u[-1],
        "-",
        marker="o",
        mfc="none",
        label=f"mpp + ssp rk3, C = {Cs[i]}",
    )
    solutions_rk4[i].rk4()
    plt.plot(
        x,
        solutions_rk4[i].u[-1],
        "--",
        marker="o",
        mfc="none",
        label=f"mpp + rk4, C = {Cs[i]}",
    )
plt.title(
    f"order {order} mpp solutions (C_max = {round(solutions[0].C_max, 4)})"
)
plt.xlabel(r"$x$")
plt.ylabel(r"$u$")
plt.legend()
plt.show()
