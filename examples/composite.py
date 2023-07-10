import numpy as np
from finite_volume.advection import AdvectionSolver
import finite_volume.plotting as plotting

u0 = "composite"
n = 256
v = 1
T = 1
courant = 0.166
order = 4
load = False

data1 = AdvectionSolver(
    u0=u0,
    n=n,
    v=v,
    T=T,
    courant=courant,
    order=order,
    apriori_limiting=True,
    aposteriori_limiting=False,
    SED=True,
    load=load,
)
data1.rk4()
print("data1")
print(f"min: {np.min(data1.u)}, max: {np.max(data1.u)}")
print()

data2 = AdvectionSolver(
    u0=u0,
    n=n,
    v=v,
    T=T,
    courant=courant,
    order=order,
    apriori_limiting=False,
    aposteriori_limiting=True,
    SED=True,
    load=load,
)
data2.rk4()
print("data2")
print(f"min: {np.min(data2.u)}, max: {np.max(data2.u)}")
print()


plotting.lineplot({"a priori": data1, "a posteriori": data2})
