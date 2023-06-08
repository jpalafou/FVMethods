import pytest
import numpy as np
from finite_volume.advection2d import AdvectionSolver

n_list = [16, 32, 64]
order_list = [1, 2, 3, 4, 5]


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
