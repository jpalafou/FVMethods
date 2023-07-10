import numpy as np
from finite_volume.advection import AdvectionSolver
import finite_volume.plotting as plotting
import matplotlib.pyplot as plt


def vortex(x, y):
    return -y, x


solver = AdvectionSolver(
    n=(256,),
    v=vortex,
    u0="disk",
    order=8,
    courant=0.8,
    x=(-1, 1),
    T=2 * np.pi,
    bc="dirichlet",
    const=0,
    apriori_limiting=True,
    aposteriori_limiting=False,
    convex=False,
    load=False,
    log_every=100000,
)
solver.rk4()

plotting.contour(solver)

plt.plot(solver.every_t, solver.min_history)
plt.xlabel("t")
plt.show()
