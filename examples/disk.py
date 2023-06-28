import numpy as np
from finite_volume.advection import AdvectionSolver
import finite_volume.plotting as plotting


def vortex(x, y):
    return -y, x


solution = AdvectionSolver(
    u0="disk",
    n=(128,),
    x=(-1, 1),
    v=vortex,
    bc="neumann",
    const=0,
    T=2 * np.pi,
    courant=0.8,
    order=4,
    flux_strategy="gauss-legendre",
    apriori_limiting=True,
    aposteriori_limiting=False,
    cause_trouble=False,
    load=True,
    adjust_time_step=False,
    modify_time_step=True,
)
solution.ssprk3()
solution.minmax()

plotting.cube(solution, k=-1)
