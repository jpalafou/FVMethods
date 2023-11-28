from finite_volume.advection import AdvectionSolver

shared_config = dict(
    u0="square",
    v=(2, 1),
    x=(0, 1),
    bc="periodic",
    const=None,
    snapshot_dt=1,
    num_snapshots=100,
    SED=False,
    PAD=(0, 1),
    courant=0.8,
    cupy=True,
    save_directory="/scratch/gpfs/jp7427/data/solutions/",
)

limiter_configs = [
    dict(
        apriori_limiting=True,
        modify_time_step=True,
    )
]

integrators = ["ssprk2", "ssprk3", "rk4"]
ps = [1, 2, 3, 4, 5, 6, 7]
ns = [32, 64, 128]
flux_quadratures = ["gauss-legendre", "transverse"]

for limiter_config in limiter_configs:
    for integrator in integrators:
        for p in ps:
            for n in ns:
                for flux_strategy in flux_quadratures:
                    solution = AdvectionSolver(
                        **shared_config,
                        **limiter_config,
                        order=p + 1,
                        n=(n,),
                        flux_strategy=flux_strategy,
                    )
                    if integrator == "ssprk2":
                        solution.ssprk2()
                    if integrator == "ssprk3":
                        solution.ssprk3()
                    if integrator == "rk4":
                        solution.rk4()
                    print(limiter_config)
                    print(f"{p=}, {n=}, {integrator=}")
                    print(solution.compute_violations()[1])
                    print(f"L2 error = {solution.periodic_error(norm='l2')}")
                    print()
