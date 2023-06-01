import numpy as np
from finite_volume.advection2d import AdvectionSolver
import matplotlib.pyplot as plt


def vortex(x, y):
    return -y, x


solution = AdvectionSolver(
    u0="disk",
    n=64,
    x=(-1, 1),
    T=2 * np.pi,
    v=vortex,
    courant=0.16,
    order=4,
    apriori_limiting="mpp",
    aposteriori_limiting=False,
    loglen=None,
)
solution.rk4()
print(f"global max: {np.max(solution.u):.2e}, global min: {np.min(solution.u):.2e}")
print(f"last max: {np.max(solution.u[-1]):.2e}, last min: {np.min(solution.u[-1]):.2e}")
solution.plot()

plt.plot(solution.t, np.amin(solution.u, axis=(1, 2)))
plt.xlabel("t")
plt.ylabel("minimum")
plt.show()
