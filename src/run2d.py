import numpy as np
import matplotlib.pyplot as plt
from util.integrate import rk4_Dt_adjust
from util.initial_condition import initial_condition2d
import util.advection2d as advection2d


# inputs
ic_type = "square"  # initial condition type
n_x = 32 # number of cells in the x direction
n_y = 32 # number of cells in the y direction
a = [2, 1]  # tranpsort speed
bounds = ((0, 1), (0, 1))  # spatial domain ((x0, x1), (y0, y1))
T = 2  # solving time

# configurations
spatial_order = 3
time_order = 3
courant_factor = 0.5
solution_scheme = "no limiter"

# x and y arrays
x_interface = np.linspace(bounds[0][0], bounds[0][1], num=n_x + 1)
x = 0.5 * (x_interface[:-1] + x_interface[1:])  # x at cell centers
h_x = (bounds[0][1] - bounds[0][0]) / n_x
y_interface = np.linspace(bounds[1][1], bounds[1][0], num=n_y + 1)
y = 0.5 * (y_interface[:-1] + y_interface[1:])  # x at cell centers
h_y = (bounds[1][1] - bounds[1][0]) / n_y
# test that arrays are exact
assert x_interface[0] == bounds[0][0]
assert x_interface[-1] == bounds[0][1]
assert y_interface[0] == bounds[1][1]
assert y_interface[-1] == bounds[1][0]

# time vector
rkorder = time_order
Dt = courant_factor * min(h_x, h_y) / max(np.abs(a))
if time_order > 4:
    rkorder = 4
    time_step_adjustment = rk4_Dt_adjust(
        min(h_x, h_y), x_bounds[1] - x_bounds[0], spatial_order
    )
    Dt = Dt * time_step_adjustment
n_time = int(np.ceil(T / Dt))
t = np.linspace(0, T, num=n_time)

# initial condition
u0 = initial_condition2d(x, y, ic_type)

# set up solution
if solution_scheme == "no limiter":
    solution = advection2d.AdvectionSolver(
        u0=u0, t=t, x=x, y=y, a=a[0], b=a[1], order=spatial_order
    )
else:
    raise BaseException(f"invalid solution scheme {solution_scheme}")
# time integration
solution.rkn(rkorder)

# plot
fig, ((ax0, ax1, ax2)) = plt.subplots(1, 3, figsize=(12, 10))
im0 = ax0.imshow(solution.u[0], extent=[bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]])
cbar0 = ax0.figure.colorbar(im0, ax=ax0, shrink=0.5)
ax0.set_title("initial condition")
im1 = ax1.imshow(solution.u[-1], extent=[bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]])
cbar1 = ax1.figure.colorbar(im1, ax=ax1, shrink=0.5)
ax1.set_title("final time step")
im2 = ax2.imshow(solution.u[-1] - solution.u[0], extent=[bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]])
cbar2 = ax2.figure.colorbar(im2, ax=ax2, shrink=0.5)
ax2.set_title("initial condition - final time step")
fig.tight_layout()
plt.show()
