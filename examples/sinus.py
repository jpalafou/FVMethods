from finite_volume.advection import AdvectionSolver
import finite_volume.plotting as plotting

u0 = "sinus"
n = (64,)
v = (1, 2)
courant = 0.8
order = 4
flux_strategy = "gauss-legendre"
load = False

detect_off = AdvectionSolver(
    u0=u0,
    n=n,
    v=v,
    courant=courant,
    order=order,
    flux_strategy=flux_strategy,
    apriori_limiting=True,
    SED=False,
    load=load,
)
detect_off.ssprk3()

detect_on = AdvectionSolver(
    u0=u0,
    n=n,
    v=v,
    courant=courant,
    order=order,
    flux_strategy=flux_strategy,
    apriori_limiting=True,
    SED=True,
    load=load,
)
detect_on.ssprk3()


plotting.lineplot(
    {"detect off ": detect_off, "detect on": detect_on}, y=0.5 + 0.5 * 1 / 64
)
