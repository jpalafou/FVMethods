import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from finite_volume.advection import AdvectionSolver

# problem initialization
u0 = "square"
bc = "periodic"
const = None
n = (128,)
x = (0, 1)
v = (2, 1)
T = 1
courant = 0.05
order = 8

# solve
no_limiter = AdvectionSolver(
    u0="sinus",
    flux_strategy="gauss-legendre",
    n=64,
    v=1,
    T=1,
    order=4,
    apriori_limiting=False,
    SED=False,
    PAD=None,
    load=False,
)
no_limiter.rk4()
no_limiter.minmax()

detect_on = AdvectionSolver(
    u0="sinus",
    flux_strategy="gauss-legendre",
    n=64,
    v=1,
    T=1,
    order=4,
    apriori_limiting=True,
    SED=True,
    PAD=(-1,1),
    load=False,
)
detect_on.rk4()
detect_on.minmax()
detect_on.rk4()

print(detect_on.u[-1] - no_limiter.u[-1])

plt.plot(no_limiter.x, no_limiter.u[-1], label='no limiter')
plt.plot(detect_on.x, detect_on.u[-1], '--', label='a priori with SED')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()