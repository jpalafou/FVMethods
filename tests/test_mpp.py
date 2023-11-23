import numpy as np
from finite_volume.advection import AdvectionSolver
import pytest

modify_time_step_configs = [
    dict(modify_time_step=False, mpp_tolerance=1e-20),
    dict(modify_time_step=True, mpp_tolerance=1e-10),
    dict(modify_time_step=True, mpp_tolerance=1e-20),
]
orders = [1, 2, 3, 4, 5, 6, 7, 8]
C_for_mpp = {1: 0.5, 2: 0.5, 3: 0.166, 4: 0.166, 5: 0.0833, 6: 0.0833, 7: 0.05, 8: 0.05}


@pytest.mark.parametrize("modify_time_step_config", modify_time_step_configs)
@pytest.mark.parametrize("order", orders)
@pytest.mark.parametrize("mpp_lite", [False, True])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("PAD", [(0, np.inf), (0, 1)])
def test_positivity1d(modify_time_step_config, order, mpp_lite, SED, PAD):
    if modify_time_step_config["modify_time_step"]:
        cfl = 0.8
    else:
        cfl = C_for_mpp[order]
    PAD = (0, np.inf)
    solution = AdvectionSolver(
        u0="composite",
        bc="periodic",
        const=None,
        n=256,
        v=1,
        snapshot_dt=1,
        num_snapshots=1,
        courant=cfl,
        order=order,
        apriori_limiting=True,
        mpp_lite=mpp_lite,
        PAD=PAD,
        **modify_time_step_config,
        load=False,
    )
    solution.rkorder()
    assert (
        -min(solution.compute_violations()[1]["worst"], 0)
        < modify_time_step_config["mpp_tolerance"]
    )


modify_time_step_configs = [
    dict(modify_time_step=False, mpp_tolerance=1e-10),
    dict(modify_time_step=True, mpp_tolerance=1e-10),
]


@pytest.mark.parametrize("modify_time_step_config", modify_time_step_configs)
@pytest.mark.parametrize("order", orders)
@pytest.mark.parametrize("mpp_lite", [False, True])
def test_mpp2d(modify_time_step_config, order, mpp_lite):
    if modify_time_step_config["modify_time_step"]:
        cfl = 0.8
    else:
        cfl = C_for_mpp[order]
    PAD = (0, 1)
    solution = AdvectionSolver(
        u0="square",
        bc="periodic",
        const=None,
        n=(128,),
        v=(2, 1),
        flux_strategy="gauss-legendre",
        snapshot_dt=1,
        num_snapshots=1,
        courant=cfl,
        order=order,
        apriori_limiting=True,
        mpp_lite=mpp_lite,
        PAD=PAD,
        **modify_time_step_config,
        load=False,
    )
    solution.rkorder()
    assert (
        -min(solution.compute_violations()[1]["worst"], 0)
        < modify_time_step_config["mpp_tolerance"]
    )
