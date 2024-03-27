from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from configs import problem_configs, solver_config
from finite_volume.advection import AdvectionSolver

limiter_configs = [
    (
        "GL apriori with SED",
        dict(
            courant=0.8,
            flux_strategy="gauss-legendre",
            apriori_limiting=True,
            SED=True,
        ),
    ),
    (
        "transverse apriori with SED",
        dict(
            courant=0.8,
            flux_strategy="transverse",
            apriori_limiting=True,
            SED=True,
        ),
    ),
]

data = []
for n, p, (limiter_config_name, limiter_config) in product(
    [32, 64, 128, 256, 512], range(8), limiter_configs
):
    solver = AdvectionSolver(
        n=(n,),
        order=p + 1,
        cupy=256,
        adjust_stepsize=6,
        snapshot_dt=1.0,
        num_snapshots=1,
        **problem_configs["sinus2d"],
        **limiter_config,
        **solver_config,
    )
    solver.rkorder(rk6=True)

    # measure error and record
    err = solver.snapshots[-1]["u"] - solver.snapshots[0]["u"]
    l1 = np.mean(np.abs(err))
    l2 = np.mean(np.square(err))
    data.append(
        dict(n=n, p=p, limiter_config=limiter_config_name, l1_err=l1, l2_err=l2)
    )
    df = pd.DataFrame(data)

    # plotting
    lc_grouped = df.groupby("limiter_config")
    for lc_name, lc_group in lc_grouped:
        fig, ax = plt.subplots()
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")

        p_grouped = lc_group.groupby("p")
        for p_name, p_group in p_grouped:
            ax.plot(
                p_group["n"],
                p_group["l1_err"],
                "-",
                marker="o",
                mfc="none",
                label=f"$p={p_name}$",
            )

        ax.legend()
        ax.set_xlabel("$n$")
        ax.set_ylabel(r"$\epsilon$")
        plt.savefig(f"experiments/images/convergence_lc_{lc_name}.png", dpi=300)
