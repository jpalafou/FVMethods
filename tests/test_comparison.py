# compare with the code from Romain's notebook

import pytest
import numpy as np
from finite_volume.advection import AdvectionSolver
from tests.simple_advection_solver import solve_high_order


def _generate_solutions(
    space_order: int,
    time_order: int,
    n: int,
    periods: int,
    cfl: float,
    apriori_lite: bool,
    smooth_extrema_detection: bool,
):
    romain_solution = solve_high_order(
        tend=periods,
        n=n,
        cfl=cfl,
        ic_type="composite",
        time=min(time_order, 4),
        space=space_order,
        limiter=apriori_lite,
        smooth_extrema_detection=smooth_extrema_detection,
    )

    my_solution = AdvectionSolver(
        num_snapshots=1,
        snapshot_dt=periods,
        n=n,
        courant=cfl,
        u0="composite",
        x=(0, 1),
        v=1,
        order=space_order,
        apriori_limiting=apriori_lite,
        mpp_lite=apriori_lite,
        SED=smooth_extrema_detection,
        load=False,
        save=False,
    )
    if time_order > 3:
        my_solution.rk4()
    elif time_order > 2:
        my_solution.rk3()
    elif time_order > 1:
        my_solution.rk2()
    else:
        my_solution.euler()
    return romain_solution, my_solution


n = 256
cfl = 0.8


@pytest.mark.parametrize("n", [128, 256, 512])
def test_initial_condition(n):
    romain_solution, my_solution = _generate_solutions(
        space_order=1,
        time_order=1,
        n=n,
        periods=1,
        cfl=cfl,
        apriori_lite=False,
        smooth_extrema_detection=False,
    )
    assert np.all(romain_solution[0] == my_solution.u_snapshots[0][1])


@pytest.mark.parametrize("n", [128, 256, 512])
def test_dt(n):
    _, my_solution = _generate_solutions(
        space_order=1,
        time_order=1,
        n=n,
        periods=1,
        cfl=cfl,
        apriori_lite=False,
        smooth_extrema_detection=False,
    )
    assert my_solution.dt == cfl / n


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("limiting", [False, True])
@pytest.mark.parametrize("smooth_extrema_detection", [False, True])
def test_solutions(order, limiting, smooth_extrema_detection):
    romain_solution, my_solution = _generate_solutions(
        space_order=order,
        time_order=4,
        n=n,
        periods=cfl / n,
        cfl=cfl,
        apriori_lite=limiting,
        smooth_extrema_detection=smooth_extrema_detection,
    )
    rmse = np.sqrt(
        np.mean(np.square(romain_solution[-1] - my_solution.u_snapshots[-1][1]))
    )
    assert rmse < 1e-16
