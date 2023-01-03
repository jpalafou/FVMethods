import numpy as np
import math
import time

# import matplotlib.pyplot as plt
from util.solve import AdvectionSolver

a = 1  # choose a positive value of the 'wind' direction

# domain
x_bounds = [-0.5, 2]
h = 0.0005
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

# solve and time 10 times
start_time = time.time()
for _ in range(10):
    AdvectionSolver(x0=u0, t=t, h=h, a=a, order=5)
print("\n" + str((time.time() - start_time) / 10) + "\n")
