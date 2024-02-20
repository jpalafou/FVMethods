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
@pytest.mark.parametrize("hancock", [False, True])
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
    hancock,
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
        hancock=hancock,
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
@pytest.mark.parametrize("hancock", [False, True])
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
    hancock,
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
        hancock=hancock,
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


@pytest.mark.parametrize(
    "initial_condition_config",
    [
        dict(
            u0="sinus",
            v=(2, 1),
            PAD=(-1, 1),
            x=(0, 1),
            snapshot_dt=1,
            bc="periodic",
            const=None,
        ),
        dict(
            u0="square",
            v=(2, 1),
            PAD=(0, 1),
            x=(0, 1),
            snapshot_dt=1,
            bc="periodic",
            const=None,
        ),
        dict(
            u0="disk",
            v=vortex,
            PAD=(0, 1),
            x=(-1, 1),
            snapshot_dt=2 * np.pi,
            bc="dirichlet",
            const=0.0,
        ),
    ],
)
@pytest.mark.parametrize(
    "limiter_config",
    [
        dict(apriori_limiting=True, mpp_lite=False),
        dict(apriori_limiting=True, mpp_lite=True),
    ],
)
@pytest.mark.parametrize("modify_time_step", [False, True])
@pytest.mark.parametrize("mpp_tolerance", [1e-10, 1e-15])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("SED", [False, True])
def test_a_priori_mpp_2d(
    initial_condition_config,
    limiter_config,
    modify_time_step,
    mpp_tolerance,
    order,
    SED,
):
    solution = AdvectionSolver(
        n=(128,),
        num_snapshots=1,
        **initial_condition_config,
        **limiter_config,
        courant=C_for_mpp[order] if not modify_time_step else 0.8,
        modify_time_step=modify_time_step,
        mpp_tolerance=mpp_tolerance,
        order=order,
        SED=SED,
        save_directory=test_directory,
    )
    solution.rkorder()
    assert solution.compute_violations()[1]["worst"] > -mpp_tolerance


@pytest.mark.parametrize(
    "initial_condition_config",
    [
        dict(
            u0="sinus",
            v=(2, 1),
            PAD=(-1, 1),
            x=(0, 1),
            snapshot_dt=1,
            bc="periodic",
            const=None,
        ),
        dict(
            u0="square",
            v=(2, 1),
            PAD=(0, 1),
            x=(0, 1),
            snapshot_dt=1,
            bc="periodic",
            const=None,
        ),
        dict(
            u0="disk",
            v=vortex,
            PAD=(0, 1),
            x=(-1, 1),
            snapshot_dt=2 * np.pi,
            bc="dirichlet",
            const=0.0,
        ),
    ],
)
@pytest.mark.parametrize("hancock", [False, True])
def test_MUSCL_mpp_2d(initial_condition_config, hancock):
    solution = AdvectionSolver(
        **initial_condition_config,
        hancock=hancock,
        n=(256,),
        num_snapshots=1,
        courant=0.5,
        order=2,
        aposteriori_limiting=True,
        fallback_limiter="PP2D",
        cause_trouble=True,
        save_directory=test_directory,
    )
    if hancock:
        solution.euler()
    else:
        solution.ssprk2()
    assert solution.compute_violations()[1]["worst"] >= -1e-10
