import numpy as np
from finite_volume.advection2d import AdvectionSolver

solution = AdvectionSolver(
    u0="square",
    n=64,
    x=(0, 1),
    v=(1, 2),
    T=1,
    courant=0.16,
    order=3,
    flux_strategy="transverse",
    # flux_strategy="gauss-legendre",
    apriori_limiting="mpp",
    aposteriori_limiting=False,
)
solution.ssprk3()
print(f"global max: {np.max(solution.u):.2e}, global min: {np.min(solution.u):.2e}")
print(
    f"final max: {np.max(solution.u[-1]):.2e}, final min: {np.min(solution.u[-1]):.2e}"
)
solution.plot()
