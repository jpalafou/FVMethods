import numpy as np
import matplotlib.pyplot as plt
from initial_condition import initial_condition
from util.advection import (
    AdvectionSolver,
    AdvectionSolver_minmod,
    AdvectionSolver_nOrder_MPP,
)

# input
ic_type = "square"  # initial condition type
a = 1  # tranpsort speed
x_bounds = [0, 1]  # spatial domain
h = 0.02  # grid size
T = 2  # solving time
Dt = 0.16 * h / a  # time step size

# array of x-values
x_interface = np.arange(x_bounds[0], x_bounds[1] + h, h)
x = 0.5 * (x_interface[:-1] + x_interface[1:])  # x at cell centers

# array of t-values
t = np.arange(0, T + Dt, Dt)

# initial values of u
u0 = initial_condition(x, ic_type)

# solutions
nolimiter_solution = AdvectionSolver(u0=u0, t=t, h=h, a=a, order=3)
nolimiter_solution.rk3()
nolimiter = nolimiter_solution.u
print("no limiter solution")
print(f"initial integral: {np.trapz(u0, x)}")
print(f"initial integral: {np.trapz(nolimiter[:, -1], x)}")
print(f"min: {min(nolimiter[:, -1])}, max: {max(nolimiter[:, -1])}")
print()

minmod_solution = AdvectionSolver_minmod(u0=u0, t=t, h=h, a=a)
minmod_solution.euler()
minmod = minmod_solution.u
print("minmod solution")
print(f"initial integral: {np.trapz(u0, x)}")
print(f"initial integral: {np.trapz(minmod[:, -1], x)}")
print(f"min: {min(minmod[:, -1])}, max: {max(minmod[:, -1])}")
print()

mpp_solution = AdvectionSolver_nOrder_MPP(u0=u0, t=t, h=h, a=a, order=3)
mpp_solution.ssp_rk3()
mpp = mpp_solution.u
print("mpp solution")
print(f"initial integral: {np.trapz(u0, x)}")
print(f"initial integral: {np.trapz(mpp[:, -1], x)}")
print(f"min: {min(mpp[:, -1])}, max: {max(mpp[:, -1])}")
print()

# plot
plt.plot(x, u0, label="initial condition")
plt.plot(x, nolimiter[:, -1], "-", marker="o", mfc="none", label="no limiter")
plt.plot(x, minmod[:, -1], "-", marker="o", mfc="none", label="minmod")
plt.plot(x, mpp[:, -1], "-", marker="o", mfc="none", label="mpp")
plt.title(f"{T} orbits of constant 1D advection" " within a periodic box")
plt.xlabel(r"$x$")
plt.ylabel(r"$u$")
plt.legend()
plt.show()
