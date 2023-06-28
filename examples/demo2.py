import numpy as np
import matplotlib.pyplot as plt
from finite_volume.advection import AdvectionSolver
import finite_volume.plotting as plotting

def vortex(x,y):
    return -y, x

u0 = 'square'
x = (0, 1)
v = (2,1)
bc = 'periodic'

T = 1
order = 4
n = (64,)
load = False

data1 = AdvectionSolver(
    u0=u0,
    T = T,
    x=x,
    v=v,
    n=n,
    order=order,
    modify_time_step=False,
    courant=0.133,
    bc = bc,
    const = 0,
    flux_strategy='gauss-legendre',
    apriori_limiting = True,
    aposteriori_limiting = False,
    smooth_extrema_detection = False,
    load = load,
)
data1.ssprk3()
data1.minmax()

data2 = AdvectionSolver(
    u0=u0,
    T = T,
    x=x,
    v=v,
    n=n,
    order=order,
    modify_time_step=True,
    courant=0.8,
    bc = bc,
    const = 0,
    flux_strategy='gauss-legendre',
    apriori_limiting = True,
    aposteriori_limiting = False,
    smooth_extrema_detection = False,
    load = load,
)
data2.ssprk3()
data2.minmax()


plotting.lineplot({
    'small time step': data1,
    'adaptive time step': data2,
    }, y = 0.5 + 1/128)