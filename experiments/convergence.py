from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from configs import limiting_schemes_2d, problem_configs, solver_config
from finite_volume.advection import AdvectionSolver

limiting_schemes = limiting_schemes_2d


data = []
for n, p, lckey in product(
    [32, 64, 128, 256, 512], range(8), ["ZS2D-convergence", "ZS2D-T-convergence"]
):
    solver = AdvectionSolver(
        n=(n,),
        order=p + 1,
        cupy=256,
        snapshot_dt=1.0,
        num_snapshots=1,
        **problem_configs["sinus2d"],
        **limiting_schemes[lckey],
        **solver_config,
    )
    solver.rkorder(rk6=True)

    # measure error and record
    err = solver.snapshots[-1]["u"] - solver.snapshots[0]["u"]
    l1 = np.mean(np.abs(err))
    l2 = np.mean(np.square(err))
    data.append(
        dict(
            n=n,
            p=p,
            scheme=lckey,
            integrator=solver.integrator_name,
            l1_err=l1,
            l2_err=l2,
            cupy=solver.cupy,
        )
    )
    df = pd.DataFrame(data)

    # plotting
    scheme_grouped = df.groupby("scheme")
    for scheme_name, scheme_group in scheme_grouped:
        fig, ax = plt.subplots()
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")

        p_grouped = scheme_group.groupby("p")
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
        plt.savefig(f"experiments/images/convergence_{scheme_name}.png", dpi=300)
