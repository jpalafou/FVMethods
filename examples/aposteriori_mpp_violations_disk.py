import numpy as np
import matplotlib.pyplot as plt
from finite_volume.advection import AdvectionSolver


def vortex(x, y):
    return -y, x


mpp_cfl = {1: 0.5, 2: 0.5, 3: 0.166, 4: 0.166, 5: 0.0833, 6: 0.0833, 7: 0.05, 8: 0.05}
u0 = "disk"
order = 4
n = (128,)
x = (-1, 1)
T = 0.5
v = vortex
bc = "dirichlet"
flux_strategy = "transverse"
convex = False

data1 = AdvectionSolver(
    u0=u0,
    n=n,
    v=v,
    x=x,
    T=T,
    bc=bc,
    order=order,
    flux_strategy=flux_strategy,
    courant=0.8,
    aposteriori_limiting=True,
    convex_aposteriori_limiting=convex,
    modify_time_step=False,
    log_every=1,
)
data1.ssprk3()
data1.minmax()

data0 = AdvectionSolver(
    u0=u0,
    n=n,
    v=v,
    x=x,
    T=T,
    bc=bc,
    order=order,
    flux_strategy="gauss-legendre",
    courant=0.8,
    aposteriori_limiting=True,
    convex_aposteriori_limiting=convex,
    modify_time_step=False,
    log_every=1,
)
data0.ssprk3()
data0.minmax()

data2 = AdvectionSolver(
    u0=u0,
    n=n,
    v=v,
    x=x,
    T=T,
    bc=bc,
    order=order,
    flux_strategy=flux_strategy,
    courant=0.8,
    aposteriori_limiting=True,
    convex_aposteriori_limiting=convex,
    modify_time_step=False,
    log_every=1,
)
data2.rk3()
data2.minmax()

data3 = AdvectionSolver(
    u0=u0,
    n=n,
    v=v,
    x=x,
    T=T,
    bc=bc,
    order=order,
    flux_strategy=flux_strategy,
    courant=0.8,
    aposteriori_limiting=True,
    convex_aposteriori_limiting=convex,
    modify_time_step=False,
    log_every=1,
)
data3.rk4()
data3.minmax()

solution_dict = {
    "ssprk3": data1,
    "ssprk3 + gauss-legendre": data0,
    "rk3": data2,
    "rk4": data3,
}

fig, axs = plt.subplots(2, 1, sharex="col", figsize=(10, 6))

colors = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "olive": "#bcbd22",
    "cyan": "#17becf",
}

color_list = list(colors.values())

if data1.ndim == 1:
    minaxis = 1
elif data1.ndim == 2:
    minaxis = (1, 2)

idx = 0
for label, solution in solution_dict.items():
    axs[0].plot(
        solution.t,
        np.maximum(np.amax(solution.u, axis=minaxis) - 1, 0),
        color=color_list[idx],
        label=label,
    )
    axs[1].plot(
        solution.t,
        np.abs(np.minimum(np.amin(solution.u, axis=minaxis), 0)),
        color=color_list[idx],
        label=label,
    )
    idx += 1

axs[0].set_yscale("log")
axs[0].set_ylim((10**-10, 10**-1))
axs[0].set_ylabel("max(u) > 1")
axs[1].set_yscale("log")
axs[1].set_ylim((10**-10, 10**-1))
axs[1].set_ylabel("|min(u) < 0|")
axs[1].set_xlabel("t")
axs[1].legend()
plt.show()
