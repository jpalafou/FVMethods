from finite_volume.advection import AdvectionSolver
import finite_volume.plotting as plotting

def vortex(x,y):
    return -y, x

u0 = 'square'
x = (0, 1)
bc = 'periodic'

order = 4
courant = 0.5
n = (64,)
load = False

print('solution 1')
data1 = AdvectionSolver(
    u0=u0,
    x=x,
    v=(1,0),
    n=n,
    order=order,
    courant=courant,
    flux_strategy='gauss-legendre',
    apriori_limiting = True,
    aposteriori_limiting = False,
    smooth_extrema_detection = False,
    NAD = None,
    PAD = None,
    cause_trouble=False,
    load = load,
)
data1.ssprk3()
data1.minmax()

print('solution 2')
data2 = AdvectionSolver(
    u0=u0,
    x=x,
    v=(1,0.25),
    n=n,
    order=order,
    courant=courant,
    flux_strategy='gauss-legendre',
    apriori_limiting = True,
    aposteriori_limiting = False,
    smooth_extrema_detection = False,
    NAD = None,
    PAD = None,
    cause_trouble=False,
    load = load,
)
data2.ssprk3()
data2.minmax()

print('solution 3')
data3 = AdvectionSolver(
    u0=u0,
    x=x,
    v=(1,0.5),
    n=n,
    order=order,
    courant=courant,
    flux_strategy='gauss-legendre',
    apriori_limiting = True,
    aposteriori_limiting = False,
    smooth_extrema_detection = False,
    NAD = None,
    PAD = None,
    cause_trouble=False,
    load = load,
)
data3.ssprk3()
data3.minmax()

print('solution 4')
data4 = AdvectionSolver(
    u0=u0,
    x=x,
    v=(1,1),
    n=n,
    order=order,
    courant=courant,
    flux_strategy='gauss-legendre',
    apriori_limiting = True,
    aposteriori_limiting = False,
    smooth_extrema_detection = False,
    NAD = None,
    PAD = None,
    cause_trouble=False,
    load = load,
)
data4.ssprk3()
data4.minmax()


plotting.minmax(
    {
    '2D a priori, v = (1, 0)': data1,
    '2D a priori, v = (1, 0.25)': data2,
    '2D a priori, v = (1, 0.5)': data3,
    '2D a priori, v = (1, 1)': data4,
    },
    # y= 0.5* 1/64 + 0.5
    )

    