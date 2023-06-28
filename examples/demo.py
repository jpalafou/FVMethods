from finite_volume.advection import AdvectionSolver
import finite_volume.plotting as plotting


def vortex(x, y):
    return -y, x


u0 = "square"
x = (0, 1)
v = (1, 2)
bc = "periodic"

order = 7
courant = 0.05
n = (64,)
load = True

data1 = AdvectionSolver(
    u0=u0,
    x=x,
    v=v,
    n=n,
    order=order,
    courant=courant,
    flux_strategy="gauss-legendre",
    apriori_limiting=True,
    aposteriori_limiting=False,
    smooth_extrema_detection=False,
    NAD=0,
    PAD=None,
    load=load,
)
data1.ssprk3()

data2 = AdvectionSolver(
    u0=u0,
    x=x,
    v=v,
    n=n,
    order=order,
    courant=courant,
    flux_strategy="transverse",
    apriori_limiting=False,
    aposteriori_limiting=True,
    smooth_extrema_detection=False,
    NAD=0,
    PAD=None,
    load=load,
)
data2.ssprk3()


plotting.minmax(
    {
        "GL + a priori": data1,
        "transvesrse + a postreriori": data2,
    }
)
