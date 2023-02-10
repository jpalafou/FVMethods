import numpy as np
import matplotlib.pyplot as plt
import time
from initial_condition import initial_condition
from util.advection import (
    AdvectionSolver_nOrder_MPP,
    AdvectionSolver_nOrder_MPP_lite,
)


# input
ic_type = "square"  # initial condition type
a = 1  # tranpsort speed
x_bounds = [0, 1]  # spatial domain
h = 0.02  # grid size
T = 50  # solving time


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
order = 7
C_high = 1.1
C_low = 0.5  # 0.049

# classic mpp, high courant
print("classic mpp, high courant\n")
solution_high = AdvectionSolver_nOrder_MPP(
    u0=u0, t=time_vector(C_high, a, h), h=h, a=a, order=order
)
start = time.time()
solution_high.rk4()
classic_time_high = time.time() - start
print(f"classic mpp solved in {classic_time_high} seconds\n")

# classic mpp, low courant
print("classic mpp, low courant\n")
solution_low = AdvectionSolver_nOrder_MPP(
    u0=u0, t=time_vector(C_low, a, h), h=h, a=a, order=order
)
start = time.time()
solution_low.rk4()
classic_time_low = time.time() - start
print(f"classic mpp solved in {classic_time_low} seconds\n")

# mpp lite, high courant
print("mpp lite, high courant\n")
solution_lite_high = AdvectionSolver_nOrder_MPP_lite(
    u0=u0, t=time_vector(C_high, a, h), h=h, a=a, order=order
)
start = time.time()
solution_lite_high.rk4()
lite_time_high = time.time() - start
print(f"mpp lite solved in {lite_time_high} seconds\n")

# mpp lite, low courant
print("mpp lite, low courant\n")
solution_lite_low = AdvectionSolver_nOrder_MPP_lite(
    u0=u0, t=time_vector(C_low, a, h), h=h, a=a, order=order
)
start = time.time()
solution_lite_low.rk4()
lite_time_low = time.time() - start
print(f"mpp lite solved in {lite_time_low} seconds\n")

# plot
plt.plot(x, u0, label="initial condition")
plt.plot(
    x,
    solution_high.u[-1],
    "-",
    marker="o",
    mfc="none",
    label=(
        f"order {order} mpp + rk4, C = {C_high}, \nsolve time "
        f"= {round(classic_time_high, 4)} seconds"
    ),
)
plt.plot(
    x,
    solution_lite_high.u[-1],
    "--",
    marker="o",
    mfc="none",
    label=(
        f"order {order} mpp lite + rk4, C = {C_high}, \nsolve "
        f"time = {round(lite_time_high, 4)} seconds"
    ),
)
plt.plot(
    x,
    solution_low.u[-1],
    "-",
    marker="o",
    mfc="none",
    label=(
        f"order {order} mpp + rk4, C = {C_low},\nsolve time "
        f"= {round(classic_time_low, 4)} seconds"
    ),
)
plt.plot(
    x,
    solution_lite_low.u[-1],
    "--",
    marker="o",
    mfc="none",
    label=(
        f"order {order} mpp lite + rk4, C = {C_low},\nsolve "
        f"time = {round(lite_time_low, 4)} seconds"
    ),
)
plt.title(f"{T} orbits of constant advection in a periodic box")
plt.xlabel(r"$x$")
plt.ylabel(r"$u$")
plt.legend()
plt.show()
