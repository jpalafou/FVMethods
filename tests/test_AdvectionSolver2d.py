import pytest
import numpy as np
import os
from finite_volume.advection import AdvectionSolver

test_directory = "data/test_solutions/"

n_list = [16, 32, 64]
order_list = [1, 2, 3, 4, 5]
C_for_mpp = {1: 0.5, 2: 0.5, 3: 0.166, 4: 0.166, 5: 0.0833}


def vortex(x, y):
    return -y, x


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


@pytest.mark.parametrize("n", [(32,), (64, 32)])
@pytest.mark.parametrize("order", [1, 2, 8])
@pytest.mark.parametrize("flux_strategy", ["gauss-legendre", "transverse"])
@pytest.mark.parametrize("apriori_limiting", [False, True])
@pytest.mark.parametrize("mpp_lite", [False, True])
@pytest.mark.parametrize("aposteriori_limiting", [False, True])
@pytest.mark.parametrize("convex", [False, True])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("NAD", [None, 0, 1e-3])
@pytest.mark.parametrize("PAD", [None, (0, 1)])
def test_init(
    n,
    order,
    flux_strategy,
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
        u0="square",
        n=n,
        v=vortex,
        order=order,
        flux_strategy=flux_strategy,
        apriori_limiting=apriori_limiting,
        mpp_lite=mpp_lite,
        aposteriori_limiting=aposteriori_limiting,
        convex=convex,
        SED=SED,
        NAD=NAD,
        PAD=PAD,
        save_directory=test_directory,
    )


@pytest.mark.parametrize("n", [(32,), (64, 32)])
@pytest.mark.parametrize("order", [1, 2, 8])
@pytest.mark.parametrize("flux_strategy", ["gauss-legendre", "transverse"])
@pytest.mark.parametrize("apriori_limiting", [False, True])
@pytest.mark.parametrize("mpp_lite", [False, True])
@pytest.mark.parametrize("aposteriori_limiting", [False, True])
@pytest.mark.parametrize("convex", [False, True])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("NAD", [None, 0, 1e-3])
@pytest.mark.parametrize("PAD", [None, (0, 1)])
def test_udot(
    n,
    order,
    flux_strategy,
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
        u0="square",
        n=n,
        v=vortex,
        order=order,
        flux_strategy=flux_strategy,
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
            n=(n,), order=order, v=(1, 2), u0="sinus", save_directory=test_directory
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
            n=(n,), order=order, v=(1, 2), u0="sinus", save_directory=test_directory
        )
        solution.rkorder()
        errorlist.append(solution.periodic_error("l1"))
    assert all(errorlist[i] - errorlist[i + 1] > 0 for i in range(len(errorlist) - 1))


@pytest.mark.parametrize("n", n_list)
@pytest.mark.parametrize("order", order_list)
def test_periodic_solution(n, order):
    """
    advecting with a velocity of (1,2) should result in the mirror of
    advecting with a velocity of (-1,-2)
    """
    forward_advection = AdvectionSolver(
        n=(n,), order=order, v=(1, 2), save_directory=test_directory
    )
    forward_advection.rkorder()
    backward_advection = AdvectionSolver(
        n=(n,), order=order, v=(-1, -2), save_directory=test_directory
    )
    backward_advection.rkorder()
    assert forward_advection.u_snapshots[-1][1] == pytest.approx(
        np.fliplr(np.flipud(backward_advection.u_snapshots[-1][1]))
    )


@pytest.mark.parametrize("order", order_list)
@pytest.mark.parametrize(
    "config",
    [
        {
            "x": (-1, 1),
            "snapshot_dt": np.pi / 4,
            "num_snapshots": 8,
            "v": vortex,
            "u0": "disk",
            "bc": "dirichlet",
            "const": 0,
        },
        {
            "x": (0, 1),
            "snapshot_dt": np.pi / 4,
            "num_snapshots": 8,
            "v": (1, 2),
            "u0": "square",
            "bc": "periodic",
            "const": None,
        },
    ],
)
def test_mpp(order, config):
    tolerance = 1e-16
    solution = AdvectionSolver(
        n=(64,),
        order=order,
        x=config["x"],
        snapshot_dt=config["snapshot_dt"],
        num_snapshots=config["num_snapshots"],
        v=config["v"],
        u0=config["u0"],
        bc=config["bc"],
        const=config["const"],
        courant=C_for_mpp[order],
        flux_strategy="gauss-legendre",
        apriori_limiting=True,
        save_directory=test_directory,
    )
    solution.ssprk3()
    assert np.min(solution.u_snapshots[-1][1]) >= 0 - tolerance
    assert np.max(solution.u_snapshots[-1][1]) <= 1 + tolerance


def test_fallback_mpp():
    ...
