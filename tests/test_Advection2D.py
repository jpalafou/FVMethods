import pytest
import numpy as np
from finite_volume.advection2d import AdvectionSolver

n_list = [16, 32, 64]
order_list = [1, 2, 3, 4, 5]
C_for_mpp = {1: 0.5, 2: 0.5, 3: 0.166, 4: 0.166, 5: 0.0833}


@pytest.mark.parametrize("order", order_list)
@pytest.mark.parametrize("flux_strategy", ["gauss-legendre", "transverse"])
@pytest.mark.parametrize("apriori_limiting", [False, True])
@pytest.mark.parametrize("aposteriori_limiting", [False, True])
@pytest.mark.parametrize("NAD", [None, 0, 1e-3])
@pytest.mark.parametrize("PAD", [None, (0, 1)])
@pytest.mark.parametrize("cause_trouble", [False, True])
def test_zero_velocity(
    order,
    flux_strategy,
    apriori_limiting,
    aposteriori_limiting,
    NAD,
    PAD,
    cause_trouble,
):
    """
    solution at t=T should be exactly the initial condition if v=(0,0)
    for any initialization
    """
    solution = AdvectionSolver(
        u0="square",
        n=64,
        x=(0, 1),
        v=(0, 0),
        T=1,
        courant=0.5,
        order=order,
        flux_strategy=flux_strategy,
        apriori_limiting=True,
        aposteriori_limiting=False,
        NAD=NAD,
        PAD=PAD,
        cause_trouble=cause_trouble,
        load=False,
    )
    solution.rk4()
    assert np.all(solution.u[-1] == solution.u[0])


@pytest.mark.parametrize("flux_strategy", ["gauss-legendre", "transverse"])
@pytest.mark.parametrize("order", order_list)
def test_meshsize_convergence(flux_strategy, order):
    """
    l1 error should decrease with finer meshes
    """
    errorlist = np.array([])
    for n in n_list:
        solution = AdvectionSolver(
            u0="sinus",
            x=(0, 1),
            v=(1, 2),
            T=1,
            courant=0.5,
            n=n,
            order=order,
            flux_strategy=flux_strategy,
            apriori_limiting=None,
            aposteriori_limiting=False,
            load=False,
        )
        solution.rkorder()
        errorlist = np.append(errorlist, solution.periodic_error("l1"))
    assert all(errorlist[:-1] - errorlist[1:] > 0)


@pytest.mark.parametrize("flux_strategy", ["gauss-legendre", "transverse"])
@pytest.mark.parametrize("n", n_list)
def test_order_convergence(flux_strategy, n):
    """
    l1 error should decrease with increasing solution order
    """
    errorlist = np.array([])
    for order in order_list:
        solution = AdvectionSolver(
            u0="sinus",
            x=(0, 1),
            v=(1, 2),
            T=1,
            courant=0.5,
            n=n,
            order=order,
            flux_strategy=flux_strategy,
            apriori_limiting=None,
            aposteriori_limiting=False,
            load=False,
        )
        solution.rkorder()
        errorlist = np.append(errorlist, solution.periodic_error("l1"))
    assert all(errorlist[:-1] - errorlist[1:] > 0)


@pytest.mark.parametrize("flux_strategy", ["gauss-legendre", "transverse"])
@pytest.mark.parametrize("order", order_list)
@pytest.mark.parametrize("n", n_list)
def test_fallback_mpp(flux_strategy, order, n):
    """
    solution should be mpp when all cells are forced to be troubled
    """
    solution = AdvectionSolver(
        u0="square",
        x=(0, 1),
        v=(1, 2),
        T=1,
        courant=0.5,
        n=n,
        order=order,
        flux_strategy=flux_strategy,
        apriori_limiting=None,
        aposteriori_limiting=True,
        cause_trouble=True,
        load=False,
    )
    solution.ssprk3()
    assert np.min(solution.u) >= 0
    assert np.max(solution.u) <= 1


@pytest.mark.parametrize("order", order_list)
@pytest.mark.parametrize("n", n_list)
def test_apriori_mpp(order, n):
    """
    solution should be mpp when apriori limiting is implemented
    """
    solution = AdvectionSolver(
        u0="square",
        x=(0, 1),
        v=(1, 2),
        T=1,
        courant=C_for_mpp[order],
        n=n,
        order=order,
        flux_strategy="gauss-legendre",
        apriori_limiting="mpp",
        aposteriori_limiting=False,
        cause_trouble=False,
        load=False,
    )
    solution.ssprk3()
    assert np.min(solution.u) >= 0 - 1e-12
    assert np.max(solution.u) <= 1 + 1e-12
