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
    courant=0.166,
    order=4,
    flux_strategy="gauss-legendre",
    apriori_limiting=True,
    aposteriori_limiting=False,
    cause_trouble=False,
    load=False,
    adjust_time_step=True,
)
solution.ssprk3()
print(f"global, min: {np.min(solution.u)}, max: {np.max(solution.u)}")
print(f" final, min: {np.min(solution.u[-1])}, max: {np.max(solution.u[-1])}")

plotting.cubeplot(solution)
