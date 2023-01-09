# solve 1D advection with a particular order of accuracy
import numpy as np
import math
import matplotlib.pyplot as plt
from util.solve import AdvectionSolver

a = 1  # choose a positive value of the 'wind' direction
order = 4

# domain
x_bounds = [0, 1]
h = 0.02
T = 2
Dt = 0.8 * h / a

# array of x-values
x_interface = np.arange(x_bounds[0], x_bounds[1] + h, h)
x = 0.5 * (x_interface[:-1] + x_interface[1:])  # x at cell centers

# array of t-values
t = np.arange(0, T + Dt, Dt)

# initial values of u
u0 = np.cos(2 * math.pi * x)

# advect for T orbits and the initial state should match the final state
# do this for many orders
plt.plot(x, u0, label="t = 0")
advection_solution = AdvectionSolver(x0=u0, t=t, h=h, a=a, order=order)
advection_solution.rk4()
u = advection_solution.x
plt.plot(
    x, u[:, -1], "--", marker="o", mfc="none", label=f"order {order} + rk4"
)
plt.title(f"{T} orbits of constant 1D advection" " within a periodic box")
plt.xlabel(r"$x$")
plt.ylabel(r"$u$")
plt.legend()
plt.show()
