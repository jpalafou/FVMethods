from finite_volume.advection import AdvectionSolver
import finite_volume.plotting as plotting


def vortex(x, y):
    return -y, x


u0 = "square"
x = (0, 1)
v = 1
bc = "periodic"

order = 4
courant = 0.5
n = 64
load = False

data1 = AdvectionSolver(
    u0=u0,
    x=x,
    v=v,
    n=n,
    order=order,
    courant=0.166,
    flux_strategy="gauss-legendre",
    apriori_limiting=True,
    aposteriori_limiting=False,
    SED=False,
    NAD=None,
    PAD=None,
    cause_trouble=False,
    load=load,
)
data1.ssprk3()


data2 = AdvectionSolver(
    u0=u0,
    x=x,
    v=v,
    n=n,
    order=order,
    courant=0.8,
    flux_strategy="gauss-legendre",
    apriori_limiting=True,
    aposteriori_limiting=False,
    SED=False,
    NAD=None,
    PAD=None,
    cause_trouble=False,
    load=load,
)
data2.ssprk3()

data3 = AdvectionSolver(
    u0=u0,
    x=x,
    v=(v, v),
    n=(n,),
    order=order,
    courant=0.166,
    flux_strategy="gauss-legendre",
    apriori_limiting=True,
    aposteriori_limiting=False,
    SED=False,
    NAD=None,
    PAD=None,
    cause_trouble=False,
    load=load,
)
data3.ssprk3()

data4 = AdvectionSolver(
    u0=u0,
    x=x,
    v=(v, v),
    n=(n,),
    order=order,
    courant=0.8,
    flux_strategy="gauss-legendre",
    apriori_limiting=True,
    aposteriori_limiting=False,
    SED=False,
    NAD=None,
    PAD=None,
    cause_trouble=False,
    load=load,
)
data4.ssprk3()


plotting.minmax(
    {
        "1D a priori, C = 0.166": data1,
        "1D a priori, C = 0.8": data2,
        "--2D a priori, C = 0.166": data3,
        "2D a priori, C = 0.8": data4,
    },
    # y= 0.5* 1/64 + 0.5
)
