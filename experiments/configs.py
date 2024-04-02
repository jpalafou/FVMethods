import numpy as np
from finite_volume.utils import dict_combinations


def vortex(x, y):
    return -y, x


problem_configs = {
    "sinus2d": dict(
        u0="sinus",
        x=(0, 1),
        y=(0, 1),
        v=(2, 1),
        PAD=(-np.inf, np.inf),
        bc="periodic",
    ),
    "composite": dict(
        u0="composite",
        x=(0, 1),
        v=1,
        PAD=(0, 1),
        bc="periodic",
    ),
    "square2d": dict(
        u0="square",
        x=(0, 1),
        y=(0, 1),
        v=(2, 1),
        PAD=(0, 1),
        bc="periodic",
    ),
    "disk": dict(
        u0="disk",
        x=(-1, 1),
        y=(-1, 1),
        v=vortex,
        PAD=(0, 1),
        bc="dirichlet",
        const={"u": 0, "trouble": 0},
    ),
}

"""
Named schemes
"""

limiting_schemes_1d = {
    "ZS": dict(apriori_limiting=True, courant="mpp", SED=True),
    "ZS-M": dict(
        apriori_limiting=True,
        mpp_lite=True,
        courant=0.8,
        adaptive_stepsize=True,
        SED=True,
    ),
    "MUSCL-Hancock": dict(
        cause_trouble=True,
        aposteriori_limiting=True,
        courant=0.8,
        hancock=True,
        fallback_limiter="minmod",
    ),
    "FM-mon": dict(
        aposteriori_limiting=True, courant=0.8, fallback_limiter="moncen", SED=True
    ),
    "FM-min-CB": dict(
        aposteriori_limiting=True,
        courant=0.8,
        fallback_limiter="minmod",
        convex=True,
        SED=True,
    ),
}

limiting_schemes_2d = {
    "ZS2D": dict(
        flux_strategy="gauss-legendre", apriori_limiting=True, courant="mpp", SED=True
    ),
    "ZS2D-T": dict(
        flux_strategy="transverse",
        apriori_limiting=True,
        mpp_lite=True,
        courant="mpp",
        SED=True,
    ),
    "ZS2D-M": dict(
        flux_strategy="gauss-legendre",
        apriori_limiting=True,
        mpp_lite=True,
        courant=0.8,
        adaptive_stepsize=True,
        SED=True,
    ),
    "ZS2D-M-Fdt": dict(
        flux_strategy="gauss-legendre",
        apriori_limiting=True,
        mpp_lite=True,
        courant=0.8,
        SED=True,
    ),
    "ZS2D-convergence": dict(
        flux_strategy="gauss-legendre",
        apriori_limiting=True,
        courant=0.8,
        adjust_stepsize=6,
        SED=True,
    ),
    "ZS2D-T-convergence": dict(
        flux_strategy="transverse",
        apriori_limiting=True,
        mpp_lite=True,
        courant=0.8,
        adjust_stepsize=6,
        SED=True,
    ),
    "MUSCL-Hancock": dict(
        cause_trouble=True,
        aposteriori_limiting=True,
        courant=0.8,
        hancock=True,
        fallback_limiter="PP2D",
    ),
    "FMH2D-min-CB": dict(
        flux_strategy="gauss-legendre",
        aposteriori_limiting=True,
        hancock=True,
        fallback_limiter="minmod",
        convex=True,
        courant=0.8,
        SED=True,
    ),
    "FMH2D-min-CB-T": dict(
        flux_strategy="transverse",
        aposteriori_limiting=True,
        hancock=True,
        fallback_limiter="minmod",
        convex=True,
        courant=0.8,
        SED=True,
    ),
    "FM2D-PP-T": dict(
        flux_strategy="transverse",
        aposteriori_limiting=True,
        fallback_limiter="PP2D",
        courant=0.8,
        SED=True,
    ),
}


"""
solving up to 1 or 100 periods
"""

limiter_configs = [
    *dict_combinations(
        dict(flux_strategy=["gauss-legendre", "transverse"], courant=[0.8])
    ),  # unlimited solver
    *dict_combinations(
        dict(
            flux_strategy=["gauss-legendre", "transverse"],
            apriori_limiting=[True],
            mpp_lite=[False, True],
            courant=[0.8],
            SED=[True],
        )
    ),  # a priori without adaptive timestep
    *dict_combinations(
        dict(
            flux_strategy=["gauss-legendre"],
            apriori_limiting=[True],
            mpp_lite=[False, True],
            courant=[0.8],
            adaptive_stepsize=[True],
            SED=[True],
        )
    ),  # a priori with adaptive timestep. Not compatible with RK6.
    *dict_combinations(
        dict(
            flux_strategy=["gauss-legendre", "transverse"],
            aposteriori_limiting=[True],
            fallback_limiter=["minmod", "moncen"],
            convex=[False, True],
            hancock=[False, True],
            fallback_to_1st_order=[False, True],
            NAD=[1e-3, 1e-5, 1e-7],
            courant=[0.8],
            SED=[True],
        )
    ),  # minmod/moncen a posteriori
    *dict_combinations(
        dict(
            flux_strategy=["gauss-legendre", "transverse"],
            aposteriori_limiting=[True],
            fallback_limiter=["PP2D"],
            convex=[False, True],
            hancock=[False, True],
            NAD=[1e-3, 1e-5, 1e-7],
            courant=[0.8],
            SED=[True],
        )
    ),  # PP2D aposteriori. Not compatible with 1D solver.
    *dict_combinations(
        dict(
            aposteriori_limiting=[True],
            cause_trouble=[True],
            hancock=[True],
            fallback_limiter=["minmod", "moncen"],
            fallback_to_1st_order=[False, True],
        )
    ),  # MUSCL-Hancock with minmod/moncen. Use euler integration.
    *dict_combinations(
        dict(
            aposteriori_limiting=[True],
            cause_trouble=[True],
            hancock=[True],
            fallback_limiter=["PP2D"],
        )
    ),  # MUSCL-Hancock with PP2D. Use euler integration.
    *dict_combinations(
        dict(
            aposteriori_limiting=[True],
            cause_trouble=[True],
            fallback_limiter=["minmod", "moncen"],
            fallback_to_1st_order=[False, True],
        )
    ),  # MUSCL with minmod/moncen. Use SSPRK2 integration.
    *dict_combinations(
        dict(
            aposteriori_limiting=[True],
            hancock=[True],
            fallback_limiter=["PP2D"],
        )
    ),  # MUSCL with PP2D. Use SSPRK2 integration.
]

solver_config = dict(
    progress_bar=True, load=True, save=True, save_directory="/scratch/gpfs/jp7427/data/"
)

"""
for computing the time per cell of each scheme
"""

timing_limiter_configs = [
    dict(
        flux_strategy="gauss-legendre",
        apriori_limiting=True,
        mpp_lite=True,
        courant=0.8,
        SED=True,
    ),  # a priori
    dict(
        flux_strategy="transverse",
        aposteriori_limiting=True,
        fallback_limiter="PP2D",
        convex=True,
        hancock=True,
        courant=0.8,
        SED=True,
    ),  # a posteriori
    dict(
        aposteriori_limiting=True,
        cause_trouble=True,
        hancock=True,
        fallback_limiter="PP2D",
    ),  # MUSCL-Hancock with PP2D
]

timing_solver_config = dict(
    progress_bar=False, save=True, save_directory="/scratch/gpfs/jp7427/data/"
)

"""
long time integration
"""

lt_limiter_configs = [
    dict(
        flux_strategy="gauss-legendre",
        apriori_limiting=True,
        mpp_lite=True,
        courant=0.8,
        adaptive_stepsize=True,
        SED=True,
    ),  # a priori with adaptive timestep. Not compatible with RK6.
    dict(
        flux_strategy="transverse",
        apriori_limiting=True,
        mpp_lite=True,
        courant=0.8,
        SED=True,
    ),  # a priori (not MPP) but less diffusive
    dict(
        flux_strategy="transverse",
        aposteriori_limiting=True,
        fallback_limiter="PP2D",
        courant=0.8,
        SED=True,
    ),  # less diffusive a posteriori
    dict(
        flux_strategy="transverse",
        aposteriori_limiting=True,
        fallback_limiter="minmod",
        hancock=True,
        convex=True,
        courant=0.8,
        SED=True,
    ),  # less MPP violating a posteriori
    dict(
        aposteriori_limiting=True,
        cause_trouble=True,
        hancock=True,
        fallback_limiter="PP2D",
    ),  # MUSCL-Hancock with PP2D. Use euler integration.
]
