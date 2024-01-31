import pytest
import numpy as np
import os
from finite_volume.advection import AdvectionSolver
from finite_volume.initial_conditions import generate_ic
from finite_volume.utils import dict_combinations

test_directory = "data/test_solutions/"


@pytest.fixture(scope="module", autouse=True)
def cleanup(request):
    """
    Remove solutions saved by tests
    """

    def remove_test_dir():
        for f in os.listdir(test_directory):
            os.remove(os.path.join(test_directory, f))
        os.rmdir(test_directory)

    request.addfinalizer(remove_test_dir)


"""
1D
"""

limiter_configs = [
    dict(),
    *dict_combinations(
        dict(
            apriori_limiting=[True],
            mpp_lite=[False, True],
            SED=[False, True],
        )
    ),
    *dict_combinations(
        dict(
            aposteriori_limiting=[True],
            fallback_limiter=["minmod", "moncen"],
            convex=[False, True],
            hancock=[False, True],
            fallback_to_first_order=[False, True],
            SED=[False, True],
        )
    ),
]


@pytest.mark.parametrize("p", range(8))
@pytest.mark.parametrize("const", [-1, 0, 1])
@pytest.mark.parametrize("ic_type", ["sinus", "square"])
@pytest.mark.parametrize("limiter_config", limiter_configs)
def test_vertical_shift_equivariance_1d(p, const, ic_type, limiter_config):
    """
    advection_solution(u(x) + const) = advection_solution(u(x)) + const
    """
    u0 = lambda x: generate_ic(type=ic_type, x=x, y=None)
    u0.__name__ = ic_type
    u0_shifted = lambda x: u0(x) + const
    u0_shifted.__name__ = ic_type + "+const"

    shared_config = dict(
        **limiter_config,
        save_directory=test_directory,
        v=1,
        n=64,
        order=p + 1,
        courant=0.8,
        snapshot_dt=1,
    )

    # baseline
    solver = AdvectionSolver(
        **shared_config,
        u0=u0,
        PAD=(0, 1),
    )
    solver.rkorder()

    # shifted initial condition
    shifted_solver = AdvectionSolver(
        **shared_config,
        u0=u0_shifted,
        PAD=(0 + const, 1 + const),
    )
    shifted_solver.rkorder()

    # check equivariance
    print(
        np.max(
            np.abs(
                solver.u_snapshots[-1][1] + const - shifted_solver.u_snapshots[-1][1]
            )
        )
    )
    assert np.all(
        np.isclose(
            solver.u_snapshots[-1][1] + const,
            shifted_solver.u_snapshots[-1][1],
            atol=1e-10,
        )
    )


@pytest.mark.parametrize("p", range(2))
@pytest.mark.parametrize("ic_type", ["sinus", "square"])
@pytest.mark.parametrize("limiter_config", limiter_configs)
def test_velocity_equivariance_1d(p, ic_type, limiter_config):
    """
    advection_solution(u(x), -v_x) = advection_solution(u(-x), v_x)
    """
    u0 = lambda x: generate_ic(type=ic_type, x=x, y=None)
    u0.__name__ = ic_type
    u0_horizontally_reflected = lambda x: np.flip(u0(x))
    u0_horizontally_reflected.__name__ = ic_type + "_horizontally_reflected"

    shared_config = dict(
        **limiter_config,
        save_directory=test_directory,
        PAD=(0, 1),
        n=64,
        order=p + 1,
        courant=0.8,
        snapshot_dt=1,
    )

    # baseline
    solver = AdvectionSolver(
        **shared_config,
        u0=u0,
        v=1,
    )
    solver.rkorder()

    # reflected
    solver_negative_velocity = AdvectionSolver(
        **shared_config,
        u0=u0_horizontally_reflected,
        v=-1,
    )
    solver_negative_velocity.rkorder()

    # check equivariance
    assert np.all(
        np.isclose(
            solver.u_snapshots[-1][1],
            np.flip(solver_negative_velocity.u_snapshots[-1][1]),
            atol=1e-10,
        )
    )
