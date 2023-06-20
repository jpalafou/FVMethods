import numpy as np
from finite_volume.advection import AdvectionSolver
import finite_volume.plotting as plotting

u0 = "square"
n = 64
v = 1
T = 1
courant = 0.5
order = 4
load = False

# data1 = AdvectionSolver(
#     u0=u0,
#     n=n,
#     v=v,
#     T=T,
#     courant=courant,
#     order=order,
#     apriori_limiting=False,
#     smooth_extrema_detection=False,
#     aposteriori_limiting=False,
#     load=load,
# )
# data1.ssprk3()
# print('data1')
# print(f"min: {np.min(data1.u)}, max: {np.max(data1.u)}")
# print()

data2 = AdvectionSolver(
    u0=u0,
    n=n,
    v=v,
    T=T,
    courant=courant,
    order=order,
    apriori_limiting=False,
    aposteriori_limiting=True,
    cause_trouble=True,
    smooth_extrema_detection=False,
    load=load,
)
data2.ssprk3()
print("data2")
print(f"min: {np.min(data2.u)}, max: {np.max(data2.u)}")
print()

data3 = AdvectionSolver(
    u0=u0,
    n=n,
    v=v,
    T=T,
    courant=courant,
    order=order,
    apriori_limiting=False,
    aposteriori_limiting=True,
    smooth_extrema_detection=False,
    load=load,
)
data3.ssprk3()
print("data3")
print(f"min: {np.min(data3.u)}, max: {np.max(data3.u)}")
print()


plotting.lineplot({"data 2": data2, "data 3": data3})
