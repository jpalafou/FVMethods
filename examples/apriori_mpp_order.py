import numpy as np
from finite_volume.advection import AdvectionSolver
import finite_volume.plotting as plotting


def vortex(x, y):
    return -y, x


order = 4
n = (64,)
v = vortex
x = (-1, 1)
T = 2 * np.pi
bc = "neumann"

data1 = AdvectionSolver(
    n=n,
    v=v,
    x=x,
    T=T,
    bc=bc,
    order=order,
    courant=0.166,
    apriori_limiting=True,
    modify_time_step=False,
    log_every=1,
)
data1.ssprk3()
data1.minmax()

data2 = AdvectionSolver(
    n=n,
    v=v,
    x=x,
    T=T,
    bc=bc,
    order=order,
    courant=0.8,
    apriori_limiting=True,
    modify_time_step=False,
    log_every=1,
)
data2.ssprk3()
data2.minmax()

data3 = AdvectionSolver(
    n=n,
    v=v,
    x=x,
    T=T,
    bc=bc,
    order=order,
    courant=0.8,
    apriori_limiting=True,
    modify_time_step=True,
    log_every=1,
)
data3.ssprk3()
data3.minmax()


plotting.minmax(
    {
        "a priori + GL + small dt": data1,
        "a priori + GL + large dt": data2,
        "a priori + GL + adaptive dt": data3,
    }
)
