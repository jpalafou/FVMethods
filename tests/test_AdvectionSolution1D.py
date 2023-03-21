import pytest
import numpy as np
from util.advection1d import AdvectionSolver


def test_meshsize_convergence():
    """
    l1 error should decrease with finer meshes
    """
    nlist = [128, 256, 512, 1024]
    errorlist = []
    for n in nlist:
        solution = AdvectionSolver(n=n, u0_preset="composite")
        solution.rkorder()
        errorlist.append(solution.find_error("l1"))
    assert all(
        errorlist[i] - errorlist[i + 1] > 0 for i in range(len(errorlist) - 1)
    )


def test_order_convergence():
    """
    l1 error should decrease with increasing order
    """
    orderlist = [1, 2, 3, 4, 5, 6]
    errorlist = []
    for order in orderlist:
        solution = AdvectionSolver(n=256, order=order, u0_preset="composite")
        solution.rkorder()
        errorlist.append(solution.find_error("l1"))
    assert all(
        errorlist[i] - errorlist[i + 1] > 0 for i in range(len(errorlist) - 1)
    )


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_periodic_solution_nolimiter(order):
    """
    advecting with a velocity of 1 should result in the mirror of
    advecting with a velocity of -1
    """
    forward_advection = AdvectionSolver(order=order, a=1)
    forward_advection.rkorder()
    backward_advection = AdvectionSolver(order=order, a=-1)
    backward_advection.rkorder()
    assert forward_advection.u[-1] == pytest.approx(
        np.flip(backward_advection.u[-1])
    )


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_mpp_limiting(order):
    """
    final solution should be bounded between 0 and 1 with mpp limitng
    """
    solution = AdvectionSolver(
        order=order, apriori="mpp", u0_preset="square", T=10
    )
    solution.rkorder()
    assert [i > 0 and i < 1 for i in solution.u[-1]]
