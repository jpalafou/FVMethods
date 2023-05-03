import numpy as np
from util.advection2d import AdvectionSolver


solution = AdvectionSolver(
    u0_preset="square",
    n=64,
    x=(0, 1),
    v=(1, 2),
    T=1,
    courant=0.16,
    order=4,
    apriori_limiting="mpp",
)
solution.rkorder()
print(
    f"global max: {np.max(solution.u):.2e}, global min: {np.min(solution.u):.2e}"
)
solution.plot()
