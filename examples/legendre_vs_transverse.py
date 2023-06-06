import numpy as np
import matplotlib.pyplot as plt
from finite_volume.advection2d import AdvectionSolver

legendre = AdvectionSolver(
    u0="square",
    n=64,
    x=(0, 1),
    v=(2, 1),
    T=1,
    courant=0.16,
    order=6,
    flux_strategy="gauss-legendre",
    apriori_limiting="mpp",
    aposteriori_limiting=False,
)
legendre.ssprk3()
print("gauss-legendre")
print(
    f"global max: {np.max(legendre.u):.2e}",
    f", global min: {np.min(legendre.u):.2e}",
)
print(
    f"final max: {np.max(legendre.u[-1]):.2e}",
    f", final min: {np.min(legendre.u[-1]):.2e}",
)
print()

transverse = AdvectionSolver(
    u0="square",
    n=64,
    x=(0, 1),
    v=(2, 1),
    T=1,
    courant=0.16,
    order=6,
    flux_strategy="transverse",
    apriori_limiting="mpp",
    aposteriori_limiting=False,
)
transverse.ssprk3()
print("transverse")
print(
    f"global max: {np.max(transverse.u):.2e}",
    f", global min: {np.min(transverse.u):.2e}",
)
print(
    f"final max: {np.max(transverse.u[-1]):.2e}",
    f", final min: {np.min(transverse.u[-1]):.2e}",
)
print()

# countour plot
plt.figure(figsize=(6, 6), dpi=150)

X, Y = np.meshgrid(transverse.x, transverse.y)
legendre_contour = plt.contour(
    X,
    Y,
    legendre.u[-1],
    levels=np.linspace(-0.15, 1.15, 13),
    colors="blue",
)
transverse_contour = plt.contour(
    X,
    Y,
    transverse.u[-1],
    levels=np.linspace(-0.15, 1.15, 13),
    colors="red",
)

# Add labels and title
plt.xlabel("x")
plt.ylabel("y")
plt.title("Periodic advection of a square T=1")

# Create custom legend labels
legend_labels = {
    "legendre_contour": "gauss-legendre",
    "transverse_contour": "transverse",
}

# Create legend handles for each contour plot
handles = []
(handle,) = plt.plot(0, 0, color=legendre_contour.get_cmap()(0))
handles.append(handle)
(handle,) = plt.plot(0, 0, color=transverse_contour.get_cmap()(0))
handles.append(handle)

# Add legend with custom labels
plt.legend(
    handles, [legend_labels["legendre_contour"], legend_labels["transverse_contour"]]
)

# Show the plot
plt.show()
