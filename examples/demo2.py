from finite_volume.advection import AdvectionSolver
import finite_volume.plotting as plotting


def vortex(x, y):
    return -y, x


data2 = AdvectionSolver(
    u0="square",
    T=1,
    x=(0, 1),
    v=(1, 1),
    n=(64,),
    order=4,
    courant=0.5,
    bc="periodic",
    const=0,
    flux_strategy="gauss-legendre",
    apriori_limiting=True,
    aposteriori_limiting=False,
    smooth_extrema_detection=True,
    visualization_tolerance=1e-4,
    NAD=1e-8,
    load=True,
    log_every=1,
)
data2.rk4()

plotting.heatmap(data2, data="theta")
