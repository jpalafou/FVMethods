import numpy as np
import matplotlib.pyplot as plt
from finite_volume.advection2d import AdvectionSolver

nolimiter = AdvectionSolver(
    u0="square",
    n=64,
    x=(0, 1),
    v=(1, 2),
    T=1,
    courant=0.16,
    order=4,
    apriori_limiting=None,
    aposteriori_limiting=False,
)
nolimiter.ssprk3()
print("no limiter")
print(
    f"global max: {np.max(nolimiter.u):.2e}",
    f", global min: {np.min(nolimiter.u):.2e}",
)
print(
    f"final max: {np.max(nolimiter.u[-1]):.2e}",
    f", final min: {np.min(nolimiter.u[-1]):.2e}",
)
print()

apriori = AdvectionSolver(
    u0="square",
    n=64,
    x=(0, 1),
    v=(1, 2),
    T=1,
    courant=0.16,
    order=4,
    apriori_limiting="mpp",
    aposteriori_limiting=False,
)
apriori.ssprk3()
print("a priori")
print(
    f"global max: {np.max(apriori.u):.2e}",
    f", global min: {np.min(apriori.u):.2e}",
)
print(
    f"final max: {np.max(apriori.u[-1]):.2e}",
    f", final min: {np.min(apriori.u[-1]):.2e}",
)
print()

aposteriori = AdvectionSolver(
    u0="square",
    n=64,
    x=(0, 1),
    v=(1, 2),
    T=1,
    courant=0.16,
    order=4,
    apriori_limiting=None,
    aposteriori_limiting=True,
)
aposteriori.ssprk3()
print("a posteriori")
print(
    f"global max: {np.max(aposteriori.u):.2e}",
    f", global min: {np.min(aposteriori.u):.2e}",
)
print(
    f"final max: {np.max(aposteriori.u[-1]):.2e}",
    f", final min: {np.min(aposteriori.u[-1]):.2e}",
)
print()

# countour plot
X, Y = np.meshgrid(apriori.x, apriori.y)
contour3 = plt.contour(
    X, Y, nolimiter.u[-1], levels=np.linspace(-0.15, 1.15, 13), colors="green"
)
contour1 = plt.contour(
    X, Y, apriori.u[-1], levels=np.linspace(-0.15, 1.15, 13), colors="blue"
)
contour2 = plt.contour(
    X, Y, aposteriori.u[-1], levels=np.linspace(-0.15, 1.15, 13), colors="red"
)

# Add labels and title
plt.xlabel("x")
plt.ylabel("y")
plt.title("Periodic advection of a square T=1")

# Show the plot
plt.show()
