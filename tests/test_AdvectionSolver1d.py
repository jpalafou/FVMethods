import os
import pytest
from finite_volume.advection import AdvectionSolver

test_directory = "data/test_solutions/"

n_list = [16, 32, 64]
order_list = [1, 2, 3, 4, 5]


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
@pytest.mark.parametrize("hancock", [False, True])
@pytest.mark.parametrize("convex", [False, True])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("NAD", [None, 0, 1e-3])
@pytest.mark.parametrize("PAD", [None, (0, 1)])
def test_init(
    order,
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
        u0="composite",
        n=32,
        v=1,
        order=order,
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


@pytest.mark.parametrize("order", [1, 2, 8])
@pytest.mark.parametrize("apriori_limiting", [False, True])
@pytest.mark.parametrize("mpp_lite", [False, True])
@pytest.mark.parametrize("aposteriori_limiting", [False, True])
@pytest.mark.parametrize("hancock", [False, True])
@pytest.mark.parametrize("convex", [False, True])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("NAD", [None, 0, 1e-3])
@pytest.mark.parametrize("PAD", [None, (0, 1)])
def test_udot(
    order,
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
        u0="composite",
        n=32,
        v=1,
        order=order,
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
    solution.one_euler_step()


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


@pytest.mark.parametrize(
    "limiter_config",
    [
        dict(apriori_limiting=True, mpp_lite=False),
        dict(apriori_limiting=True, mpp_lite=True),
    ],
)
@pytest.mark.parametrize("adaptive_stepsize", [False, True])
@pytest.mark.parametrize("mpp_tolerance", [1e-15])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("SED", [False, True])
def test_a_priori_mpp_1d(limiter_config, adaptive_stepsize, mpp_tolerance, order, SED):
    solution = AdvectionSolver(
        u0="composite",
        bc="periodic",
        PAD=(0, 1),
        const=None,
        n=256,
        v=1,
        snapshot_dt=1,
        num_snapshots=1,
        courant="mpp" if not adaptive_stepsize else 0.8,
        adaptive_stepsize=adaptive_stepsize,
        mpp_tolerance=mpp_tolerance,
        **limiter_config,
        SED=SED,
        order=order,
        save_directory=test_directory,
    )
    solution.rkorder()
    assert solution.compute_mpp_violations()[1]["worst"] > -mpp_tolerance


@pytest.mark.parametrize(
    "IC_PAD", [("composite", (0, 1)), ("square", (0, 1)), ("sinus", (-1, 1))]
)
@pytest.mark.parametrize("hancock", [False, True])
@pytest.mark.parametrize("fallback_limiter", ["minmod", "moncen"])
@pytest.mark.parametrize("fallback_to_1st_order", [False, True])
@pytest.mark.parametrize("SED", [False, True])
def test_MUSCL_mpp_1d(IC_PAD, hancock, fallback_limiter, fallback_to_1st_order, SED):
    IC, PAD = IC_PAD
    solution = AdvectionSolver(
        u0=IC,
        bc="periodic",
        PAD=PAD,
        const=None,
        n=256,
        v=1,
        snapshot_dt=1,
        num_snapshots=1,
        courant=0.8,
        mpp_tolerance=1e-10,
        order=2,
        aposteriori_limiting=True,
        hancock=hancock,
        fallback_limiter=fallback_limiter,
        fallback_to_1st_order=fallback_to_1st_order,
        cause_trouble=True,
        SED=SED,
        save_directory=test_directory,
    )
    if hancock:
        solution.euler()
    else:
        solution.ssprk2()
    assert solution.compute_mpp_violations()[1]["worst"] >= 0.0
