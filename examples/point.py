import numpy as np
import matplotlib.pyplot as plt
from finite_volume.advection2d import AdvectionSolver

def ic(x,y):
    m, n = len(y), len(x)
    i = int(m / 2)
    j = int(n / 2)
    u = np.zeros((m, n))
    u[i, j] = 1.
    return u

solution = AdvectionSolver(
    u0="sinus",
    n=64,
    x=(0, 1),
    v=(1, 2),
    T=1,
    courant=0.16,
    order=4,
    flux_strategy="transverse",
    apriori_limiting="mpp",
    aposteriori_limiting=False,
)
solution.euler()


fig, ax = plt.subplots(figsize=(6, 6))
ind = 1
pos = ax.imshow(solution.u[ind], cmap='hot', interpolation='none')
fig.colorbar(pos, ax=ax)
plt.title("solution after one timestep")
print(f"min: {np.min(solution.u[ind])}")
print(f"max: {np.max(solution.u[ind])}")
plt.show()