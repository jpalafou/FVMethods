import numpy as np
from finite_volume.advection import AdvectionSolver
import finite_volume.plotting as plotting


def vortex(x, y):
    return -y, x


mpp_cfl = {1: 0.5, 2: 0.5, 3: 0.166, 4: 0.166, 5: 0.0833, 6: 0.0833, 7: 0.05, 8: 0.05}
n = (64,)
v = vortex
x = (-1, 1)
T = 2 * np.pi
bc = "dirichlet"

solution_dictionary = {}

for order in [4, 6, 8]:
    data1 = AdvectionSolver(
        n=n,
        v=v,
        x=x,
        T=T,
        bc=bc,
        order=order,
        courant=mpp_cfl[8],
        apriori_limiting=True,
        modify_time_step=False,
        log_every=1,
    )
    data1.ssprk3()
    solution_dictionary["order " + str(order) + ", small dt"] = data1
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
        modify_time_step=True,
        log_every=1,
    )
    data2.ssprk3()
    solution_dictionary["order " + str(order) + ", adaptive dt"] = data2
    data2.minmax()

plotting.minmax(solution_dictionary)
