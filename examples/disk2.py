import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from finite_volume.advection import AdvectionSolver


# problem initialization
def vortex(x, y):
    return -y, x


u0 = "disk plus hill"
bc = "dirichlet"
const = 0
n = (64,)
x = (-1, 1)
v = vortex
T = 2 * np.pi
courant = 0.05
order = 8

# solve
data1 = AdvectionSolver(
    u0=u0,
    bc=bc,
    const=const,
    n=n,
    x=x,
    v=v,
    T=T,
    courant=0.8,
    modify_time_step=True,
    order=order,
    flux_strategy="gauss-legendre",
    apriori_limiting=True,
    mpp_lite=True,
    aposteriori_limiting=False,
    convex=False,
    SED=True,
    load=False,
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
    T=T,
    courant=0.8,
    modify_time_step=False,
    order=order,
    flux_strategy="transverse",
    apriori_limiting=False,
    mpp_lite=False,
    aposteriori_limiting=True,
    convex=True,
    SED=True,
    load=False,
)
data2.rk4()
print("data2")
data2.minmax()

plt.plot(data1.y, data1.u[0][:, int(n[0] / 2)], "k-")
plt.plot(data1.y, data1.u[-1][:, int(n[0] / 2)], "--")
plt.plot(data2.y, data2.u[-1][:, int(n[0] / 2)], "--")
plt.show()

# plot
plt.gca().set_aspect("equal")
X, Y = np.meshgrid(data1.x, data1.y)
contour1 = plt.contour(
    X, Y, data1.u[-1], levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors="tab:blue"
)
contour1.collections[0].set_label("data1")
contour2 = plt.contour(
    X,
    Y,
    data2.u[-1],
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
