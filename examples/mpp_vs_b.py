from numpy import sqrt
from finite_volume.advection import AdvectionSolver
import finite_volume.plotting as plotting

u0 = "square"
order = 4
n = (64,)
x = (0, 1)
T = 1
bc = "periodic"
courant = 0.8

data1 = AdvectionSolver(
    u0=u0,
    n=n,
    v=(2, 1),
    x=x,
    T=T,
    bc=bc,
    order=order,
    courant=courant,
    apriori_limiting=True,
    modify_time_step=False,
    log_every=1,
)
data1.ssprk3()
data1.minmax()

data0 = AdvectionSolver(
    u0=u0,
    n=64,
    v=1,
    x=x,
    T=T,
    bc=bc,
    order=order,
    courant=courant,
    apriori_limiting=True,
    modify_time_step=False,
    log_every=1,
)
data0.ssprk3()
data0.minmax()

data2 = AdvectionSolver(
    u0=u0,
    n=n,
    v=(4 / 5, 3 / 5),
    x=x,
    T=T,
    bc=bc,
    order=order,
    courant=courant,
    apriori_limiting=True,
    modify_time_step=False,
    log_every=1,
)
data2.ssprk3()
data2.minmax()

data3 = AdvectionSolver(
    u0=u0,
    n=n,
    v=(sqrt(2) / 2, sqrt(2) / 2),
    x=x,
    T=T,
    bc=bc,
    order=order,
    courant=courant,
    apriori_limiting=True,
    modify_time_step=False,
    log_every=1,
)
data3.ssprk3()
data3.minmax()


plotting.minmax(
    {
        "v = 1 (1D)": data0,
        "v = (1, 0)": data1,
        "v = (0.75, 0.25)": data2,
        "v = (sqrt(2), sqrt(2))": data3,
    },
    # y= 0.5* 1/64 + 0.5
)
