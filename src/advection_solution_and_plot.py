import numpy as np
import math
import matplotlib.pyplot as plt
from util.solve import AdvectionSolver

a = 1  # choose a positive value of the 'wind' direction

# domain
x_bounds = [-0.5, 2]
h = 0.005
T = 5
Dt = 0.8 * h / a

# array of x-values
x = np.arange(x_bounds[0], x_bounds[1] + h, h)

# array of t-values
t = np.arange(0, T + Dt, Dt)

# initial values of u
u0 = np.array(
    [np.cos(5 * i) if i > -math.pi / 10 and i < math.pi / 10 else 0 for i in x]
)

# the advection speed is 1 and the domain length matches the time, so the last
# state should match the initial state
plt.plot(x, u0, label="t = 0")
for order in range(1, 8):
    advection_solution = AdvectionSolver(x0=u0, t=t, h=h, a=a, order=order)
    advection_solution.rk4()
    u = advection_solution.x
    plt.plot(x, u[:, -1], label=f"order {order}")
plt.title(
    f"{round(T / (abs(x[-1] - x[0]) / a), 2)} orbits of constant 1D advection"
    " within a periodic box"
)
plt.xlabel(r"$x$")
plt.ylabel(r"$u$")
plt.legend()
plt.show()
