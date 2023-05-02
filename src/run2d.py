import warnings
import numpy as np
from util.advection2d import AdvectionSolver

warnings.filterwarnings("ignore")


def a(x, y):
    """
    args:
        1d arrays x and y
    returns:
        tuple of meshes of both components of velocity in x, y
        for a vortex advection field
    """
    xx, yy = np.meshgrid(x, y)
    vx, vy = -yy, xx
    return vx, vy


solution = AdvectionSolver(
    u0_preset="disk",
    n=128,
    x=(-1, 1),
    y=(-1, 1),
    T=2 * np.pi,
    a=a,
    courant=0.5,
    order=4,
)
solution.rkorder()
solution.plot()
