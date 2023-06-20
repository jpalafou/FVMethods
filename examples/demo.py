from finite_volume.advection import AdvectionSolver

# import finite_volume.plotting as plotting


solution = AdvectionSolver(
    u0="square",
    courant=0.8,
    order=4,
    flux_strategy="gauss-legendre",
    x=(0, 1),
    T=1,
    n=(8,),
    v=(1, 2),
    apriori_limiting=False,
    smooth_extrema_detection=True,
    aposteriori_limiting=True,
    load=False,
)
solution.one_euler_step()
