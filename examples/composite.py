import matplotlib.pyplot as plt
from finite_volume.advection import AdvectionSolver

# problem initialization
u0 = "composite"
bc = "periodic"
const = None
n = 256
v = 1
T = 1
courant = 0.05
order = 8
flux_strategy = "gauss"

# solve
data1 = AdvectionSolver(
    u0=u0,
    bc=bc,
    const=const,
    n=n,
    v=v,
    T=T,
    courant=0.8,
    modify_time_step=True,
    order=order,
    flux_strategy=flux_strategy,
    apriori_limiting=True,
    mpp_lite=True,
    aposteriori_limiting=False,
    convex=False,
    NAD=None,
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
    v=v,
    T=T,
    courant=0.8,
    modify_time_step=False,
    order=order,
    flux_strategy=flux_strategy,
    apriori_limiting=False,
    mpp_lite=False,
    aposteriori_limiting=True,
    convex=True,
    NAD=1e-10,
    SED=True,
    load=False,
)
data2.rk4()
print("data2")
data2.minmax()

# plot
plt.plot(data1.x, data1.u[0], "k-")
plt.plot(data1.x, data1.u[-1], "o--", label="data1", markerfacecolor="none")
plt.plot(data2.x, data2.u[-1], "o--", label="data2", markerfacecolor="none")
plt.xlabel("x")
plt.ylabel(r"$\bar{u}$")
plt.legend()
plt.show()
