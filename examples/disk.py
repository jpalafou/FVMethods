import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from finite_volume.advection import AdvectionSolver


# problem initialization
def vortex(x, y):
    return -y, x


u0 = "disk"
bc = "dirichlet"
const = 0
n = (64,)
x = (-1, 1)
v = vortex
snapshot_dt = np.pi / 4
num_snapshots = 8
T = 2 * np.pi
courant = 0.8
order = 8

# solve
data1 = AdvectionSolver(
    u0=u0,
    bc=bc,
    const=const,
    n=n,
    x=x,
    v=v,
    snapshot_dt=snapshot_dt,
    num_snapshots=num_snapshots,
    courant=courant,
    order=order,
    flux_strategy="gauss-legendre",
    apriori_limiting=True,
    mpp_lite=True,
    aposteriori_limiting=False,
    convex=False,
    SED=False,
    load=True,
)
data1.rk4()
print("data1")
data1.minmax()

data2 = AdvectionSolver(
    u0=u0,
    bc=bc,
    const=const,
    n=n,
    x=x,
    v=v,
    snapshot_dt=snapshot_dt,
    num_snapshots=num_snapshots,
    courant=courant,
    order=order,
    flux_strategy="transverse",
    apriori_limiting=False,
    mpp_lite=False,
    aposteriori_limiting=True,
    convex=True,
    SED=False,
    load=True,
)
data2.rk4()
print("data2")
data2.minmax()

# plot
plt.gca().set_aspect("equal")
X, Y = np.meshgrid(data1.x, data1.y)
contour1 = plt.contour(
    X, Y, data1.u_snapshots[-1][1], levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors="tab:blue"
)
contour1.collections[0].set_label("data1")
contour2 = plt.contour(
    X,
    Y,
    data2.u_snapshots[-1][1],
    levels=[0.1, 0.3, 0.5, 0.7, 0.9],
    linestyles="dashed",
    colors="tab:orange",
)
contour2.collections[0].set_label("data2")

# Create proxy artists for the legend
blue_line = mlines.Line2D([], [], color="tab:blue", label="Dataset 1")
orange_dashed_line = mlines.Line2D(
    [], [], color="tab:orange", linestyle="dashed", label="Dataset 2"
)

plt.xlabel("x")
plt.ylabel("y")
plt.legend(handles=[blue_line, orange_dashed_line])
plt.show()
