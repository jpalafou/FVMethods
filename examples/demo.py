from finite_volume.advection import AdvectionSolver
import finite_volume.plotting as plotting
import matplotlib.pyplot as plt

solver = AdvectionSolver(
    n=(64,),
    v=(2, 1),
    u0="square",
    order=4,
    courant=0.8,
    aposteriori_limiting=True,
    convex_aposteriori_limiting=True,
    load=True,
    log_every=100000,
)
solver.rk4()

plotting.contour(solver)

plt.plot(solver.every_t, solver.min_history)
plt.xlabel("t")
plt.show()
