# solve 1D advection using schemes of varying order, plot to compare
import numpy as np
import math
import matplotlib.pyplot as plt
from util.solve import AdvectionSolver


# inputs
ic_type = "square"  # initial condition type
a = 1  # tranpsort speed
x_bounds = [0, 1]  # spatial domain
h = 0.005  # grid size
T = 2  # solving time
Dt = 0.8 * h / a  # time step size

# array of x-values
x_interface = np.arange(x_bounds[0], x_bounds[1] + h, h)
x = 0.5 * (x_interface[:-1] + x_interface[1:])  # x at cell centers

# array of t-values
t = np.arange(0, T + Dt, Dt)

# initial values of u
if ic_type == "sinus":
    u0 = np.cos(2 * math.pi * x)
elif ic_type == "square":
    u0 = np.array(
        [np.heaviside(i - 0.25, 1) - np.heaviside(i - 0.75, 1) for i in x]
    )
elif ic_type == "composite":
    u0 = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] >= 0.1 and x[i] <= 0.2:
            u0[i] = (
                1
                / 6
                * (
                    np.exp(
                        -np.log(2)
                        / 36
                        / 0.0025**2
                        * (x[i] - 0.0025 - 0.15) ** 2
                    )
                    + np.exp(
                        -np.log(2)
                        / 36
                        / 0.0025**2
                        * (x[i] + 0.0025 - 0.15) ** 2
                    )
                    + 4
                    * np.exp(
                        -np.log(2) / 36 / 0.0025**2 * (x[i] - 0.15) ** 2
                    )
                )
            )
        if x[i] >= 0.3 and x[i] <= 0.4:
            u0[i] = 0.75
        if x[i] >= 0.5 and x[i] <= 0.6:
            u0[i] = 1 - abs(20 * (x[i] - 0.55))
        if x[i] >= 0.7 and x[i] <= 0.8:
            u0[i] = (
                1
                / 6
                * (
                    np.sqrt(max(1 - (20 * (x[i] - 0.75 - 0.0025)) ** 2, 0))
                    + np.sqrt(max(1 - (20 * (x[i] - 0.75 + 0.0025)) ** 2, 0))
                    + 4 * np.sqrt(max(1 - (20 * (x[i] - 0.75)) ** 2, 0))
                )
            )

# solve
# order 1
advection_solution = AdvectionSolver(x0=u0, t=t, h=h, a=a, order=1)
advection_solution.euler()
order1 = advection_solution.x
# order 2
advection_solution = AdvectionSolver(x0=u0, t=t, h=h, a=a, order=2)
advection_solution.rk2()
order2 = advection_solution.x
# order 3
advection_solution = AdvectionSolver(x0=u0, t=t, h=h, a=a, order=3)
advection_solution.rk3()
order3 = advection_solution.x
# order 4
advection_solution = AdvectionSolver(x0=u0, t=t, h=h, a=a, order=4)
advection_solution.rk4()
order4 = advection_solution.x

# plot
plt.plot(x, u0, label="initial condition")
plt.plot(
    x, order1[:, -1], "--", marker="o", mfc="none", label="order 1 + euler"
)
plt.plot(x, order2[:, -1], "--", marker="o", mfc="none", label="order 2 + rk2")
# second order might look weird because no Hancock fix
plt.plot(x, order3[:, -1], "--", marker="o", mfc="none", label="order 3 + rk3")
plt.plot(x, order4[:, -1], "--", marker="o", mfc="none", label="order 4 + rk4")
plt.title(f"{T} orbits of constant 1D advection" " within a periodic box")
plt.xlabel(r"$x$")
plt.ylabel(r"$u$")
plt.legend()
plt.show()
