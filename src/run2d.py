import warnings
from util.advection2d import AdvectionSolver

warnings.filterwarnings("ignore")

solution = AdvectionSolver(
    u0_preset="square",
    n=(32, 32),
    x=(0, 1),
    y=(0, 1),
    T=2,
    a=(1, 1),
    courant=0.5,
    order=4,
)
solution.rkorder()
solution.plot_error()
