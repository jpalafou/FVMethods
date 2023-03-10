import warnings
import util.advection2d as advection2d

warnings.filterwarnings("ignore")

solution = advection2d.AdvectionSolver(
    u0_preset="sinus",
    n=(32, 32),
    x=(0, 1),
    y=(0, 1),
    T=2,
    a=(1, 1),
    courant=0.5,
    order=4,
    posteriori=False,
)
solution.rkn(solution.order)
solution.plot_error()
