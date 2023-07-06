from finite_volume.advection import AdvectionSolver
import finite_volume.plotting as plotting
import matplotlib.pyplot as plt


solver = AdvectionSolver(
    n=(128,),
    v=(2, 1),
    u0="square",
    order=8,
    courant=0.05,
    x=(0, 1),
    T=1,
    apriori_limiting=True,
    aposteriori_limiting=False,
    convex_aposteriori_limiting=False,
    load=False,
    log_every=100000,
)
solver.rk4()

plotting.contour(solver)

plt.plot(solver.every_t, solver.min_history)
plt.xlabel("t")
plt.show()
