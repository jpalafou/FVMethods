import numpy as np
from finite_volume.advection import AdvectionSolver
import pytest

modify_time_step_configs = [
    dict(modify_time_step=True, tol=1e-16),
    dict(modify_time_step=True, tol=0),
]
orders = [1, 2, 3, 4, 5, 6, 7, 8]
C_for_mpp = {1: 0.5, 2: 0.5, 3: 0.166, 4: 0.166, 5: 0.0833, 6: 0.0833, 7: 0.05, 8: 0.05}
test_tolerance = 1e-17


@pytest.mark.parametrize("modify_time_step_config", modify_time_step_configs)
@pytest.mark.parametrize("order", orders)
@pytest.mark.parametrize("mpp_lite", [False, True])
@pytest.mark.parametrize("config", [dict(SED=False), dict(SED=True, PAD=(0, np.inf))])
def test_positivity1d(modify_time_step_config, order, mpp_lite, config):
    if modify_time_step_config["modify_time_step"]:
        cfl = 0.8
    else:
        cfl = C_for_mpp[order]
    solution = AdvectionSolver(
        u0="composite",
        bc="periodic",
        const=None,
        n=256,
        v=1,
        snapshot_dt=1,
        num_snapshots=1,
        courant=cfl,
        modify_time_step=modify_time_step_config["modify_time_step"],
        mpp_tolerance=modify_time_step_config["tol"],
        order=order,
        apriori_limiting=True,
        mpp_lite=mpp_lite,
        **config,
        load=False,
    )
    solution.rkorder()
    assert solution.abs_min >= solution.maximum_princicple[0] - test_tolerance


@pytest.mark.parametrize("modify_time_step_config", modify_time_step_configs)
@pytest.mark.parametrize("order", orders)
@pytest.mark.parametrize("mpp_lite", [False, True])
def test_mpp2d(modify_time_step_config, order, mpp_lite):
    if modify_time_step_config["modify_time_step"]:
        cfl = 0.8
    else:
        cfl = C_for_mpp[order]
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
        modify_time_step=modify_time_step_config["modify_time_step"],
        mpp_tolerance=modify_time_step_config["tol"],
        order=order,
        apriori_limiting=True,
        mpp_lite=mpp_lite,
        load=False,
    )
    solution.rkorder()
    assert solution.abs_min >= solution.maximum_princicple[0] - test_tolerance
    assert solution.abs_max <= solution.maximum_princicple[1] + test_tolerance
