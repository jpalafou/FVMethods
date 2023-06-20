from finite_volume.advection import AdvectionSolver
import finite_volume.plotting as plotting

u0 = "square"
n = (32,)
v = (1, 2)
courant = 0.166
order = 4
flux_strategy = "gauss-legendre"
load = False

data1 = AdvectionSolver(
    u0=u0,
    n=n,
    v=v,
    courant=courant,
    order=order,
    flux_strategy=flux_strategy,
    apriori_limiting=True,
    aposteriori_limiting=False,
    smooth_extrema_detection=True,
    load=load,
)
data1.ssprk3()

data2 = AdvectionSolver(
    u0=u0,
    n=n,
    v=v,
    courant=courant,
    order=order,
    flux_strategy=flux_strategy,
    apriori_limiting=False,
    aposteriori_limiting=True,
    smooth_extrema_detection=True,
    load=load,
)
data2.ssprk3()


plotting.lineplot({"a priori": data1, "a posteriori": data2}, y=0.5 + 0.5 * 1 / 32)
