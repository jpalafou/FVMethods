from finite_volume.advection import AdvectionSolver

solver = AdvectionSolver(
    n=256,
    v=1,
    u0="composite",
    order=4,
    aposteriori_limiting=True,
    convex_aposteriori_limiting=True,
    load=False,
)
solver.one_euler_step()
