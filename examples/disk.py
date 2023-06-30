import numpy as np
from finite_volume.advection import AdvectionSolver
import finite_volume.plotting as plotting


def vortex(x, y):
    return -y, x


solution = AdvectionSolver(
    u0="disk",
    n=(64,),
    x=(-1, 1),
    v=vortex,
    bc="neumann",
    const=0,
    T=2 * np.pi,
    courant=0.166,
    order=4,
    flux_strategy="gauss-legendre",
    apriori_limiting=True,
    load=True,
    modify_time_step=False,
)
solution.ssprk3()
solution.minmax()

plotting.cube(solution)

fast_solution = AdvectionSolver(
    u0="disk",
    n=(64,),
    x=(-1, 1),
    v=vortex,
    bc="neumann",
    const=0,
    T=2 * np.pi,
    courant=0.8,
    order=4,
    flux_strategy="gauss-legendre",
    apriori_limiting=True,
    load=True,
    modify_time_step=True,
)
fast_solution.ssprk3()
fast_solution.minmax()


plotting.minmax({"C=0.166": solution, "dt refinement": fast_solution})
