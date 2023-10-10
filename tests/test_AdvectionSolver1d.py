import pytest
import numpy as np
import os
from finite_volume.advection import AdvectionSolver

test_directory = "data/test_solutions/"

n_list = [16, 32, 64]
order_list = [1, 2, 3, 4, 5]
C_for_mpp = {1: 0.5, 2: 0.5, 3: 0.166, 4: 0.166, 5: 0.0833}


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


@pytest.mark.parametrize("order", [1, 2, 8])
@pytest.mark.parametrize("apriori_limiting", [False, True])
@pytest.mark.parametrize("mpp_lite", [False, True])
@pytest.mark.parametrize("aposteriori_limiting", [False, True])
@pytest.mark.parametrize("convex", [False, True])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("NAD", [None, 0, 1e-3])
@pytest.mark.parametrize("PAD", [None, (0, 1)])
def test_init(
    order,
    apriori_limiting,
    mpp_lite,
    aposteriori_limiting,
    convex,
    SED,
    NAD,
    PAD,
):
    """
    initialization the solver shouldn't return an error
    """
    AdvectionSolver(
        u0="composite",
        n=32,
        v=1,
        order=order,
        apriori_limiting=apriori_limiting,
        mpp_lite=mpp_lite,
        aposteriori_limiting=aposteriori_limiting,
        convex=convex,
        SED=SED,
        NAD=NAD,
        PAD=PAD,
        save_directory=test_directory,
    )


@pytest.mark.parametrize("order", [1, 2, 8])
@pytest.mark.parametrize("apriori_limiting", [False, True])
@pytest.mark.parametrize("mpp_lite", [False, True])
@pytest.mark.parametrize("aposteriori_limiting", [False, True])
@pytest.mark.parametrize("convex", [False, True])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("NAD", [None, 0, 1e-3])
@pytest.mark.parametrize("PAD", [None, (0, 1)])
def test_udot(
    order,
    apriori_limiting,
    mpp_lite,
    aposteriori_limiting,
    convex,
    SED,
    NAD,
    PAD,
):
    """
    evaluating udot shouldn't return an error
    """
    solution = AdvectionSolver(
        u0="composite",
        n=32,
        v=1,
        order=order,
        apriori_limiting=apriori_limiting,
        mpp_lite=mpp_lite,
        aposteriori_limiting=aposteriori_limiting,
        convex=convex,
        SED=SED,
        NAD=NAD,
        PAD=PAD,
        save_directory=test_directory,
    )
    solution.udot(solution.u_snapshots[0][1], t=solution.timestamps[0], dt=solution.dt)


@pytest.mark.parametrize("n", n_list)
def test_order_convergence(n):
    """
    l1 error should decrease with increasing order
    """
    errorlist = []
    for order in order_list:
        solution = AdvectionSolver(
            n=n, order=order, v=1, u0="sinus", save_directory=test_directory
        )
        solution.rkorder()
        errorlist.append(solution.periodic_error("l1"))
    assert all(errorlist[i] - errorlist[i + 1] > 0 for i in range(len(errorlist) - 1))


@pytest.mark.parametrize("order", order_list)
def test_mesh_convergence(order):
    """
    l1 error should decrease with increasing order
    """
    errorlist = []
    for n in n_list:
        solution = AdvectionSolver(
            n=n, order=order, v=1, u0="sinus", save_directory=test_directory
        )
        solution.rkorder()
        errorlist.append(solution.periodic_error("l1"))
    assert all(errorlist[i] - errorlist[i + 1] > 0 for i in range(len(errorlist) - 1))


@pytest.mark.parametrize("n", n_list)
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_periodic_solution(n, order):
    """
    advecting with a velocity of 1 should result in the mirror of
    advecting with a velocity of -1
    """
    forward_advection = AdvectionSolver(
        n=n, order=order, v=1, save_directory=test_directory
    )
    forward_advection.rkorder()
    backward_advection = AdvectionSolver(
        n=n, order=order, v=-1, save_directory=test_directory
    )
    backward_advection.rkorder()
    assert forward_advection.u_snapshots[-1][1] == pytest.approx(
        np.flip(backward_advection.u_snapshots[-1][1])
    )


@pytest.mark.parametrize("order", order_list)
@pytest.mark.parametrize(
    "config",
    [
        {"n": 64, "u0": "square"},
        {"n": 256, "u0": "composite"},
    ],
)
def test_apriori_mpp(order, config):
    tolerance = 1e-16
    solution = AdvectionSolver(
        n=config["n"],
        order=order,
        v=1,
        u0=config["u0"],
        courant=C_for_mpp[order],
        apriori_limiting=True,
        save_directory=test_directory,
    )
    solution.ssprk3()
    assert np.min(solution.u_snapshots[-1][1]) >= 0 - tolerance
    assert np.max(solution.u_snapshots[-1][1]) <= 1 + tolerance


@pytest.mark.parametrize("order", order_list)
@pytest.mark.parametrize(
    "config",
    [
        {"n": 64, "u0": "square"},
        {"n": 256, "u0": "composite"},
    ],
)
def test_fallback_mpp(order, config):
    tolerance = 1e-16
    solution = AdvectionSolver(
        n=config["n"],
        order=order,
        v=1,
        u0=config["u0"],
        courant=0.5,
        aposteriori_limiting=True,
        cause_trouble=True,
        save_directory=test_directory,
    )
    solution.ssprk3()
    assert np.min(solution.u_snapshots[-1][1]) >= 0 - tolerance
    assert np.max(solution.u_snapshots[-1][1]) <= 1 + tolerance
