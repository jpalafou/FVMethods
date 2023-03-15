import pytest
import numpy as np
from util.advection1d import AdvectionSolver


def test_meshsize_convergence():
    """
    l1 error should decrease with finer meshes
    """
    nlist = [32, 64, 128, 256, 512]
    errorlist = []
    for n in nlist:
        solution = AdvectionSolver(n=n)
        solution.euler()
        errorlist.append(np.sum(np.abs(solution.u[-1] - solution.u[0])) / n)
    assert [
        errorlist[i + 1] - errorlist[i] > 0 for i in range(len(errorlist) - 1)
    ]


def test_order_convergence():
    """
    l1 error should decrease with increasing order
    """
    orderlist = [1, 2, 3, 4, 5]
    errorlist = []
    for order in orderlist:
        solution = AdvectionSolver(order=order)
        solution.rkn(order)
        errorlist.append(
            np.sum(np.abs(solution.u[-1] - solution.u[0])) / len(solution.u[0])
        )
    assert [
        errorlist[i + 1] - errorlist[i] > 0 for i in range(len(errorlist) - 1)
    ]


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_periodic_solution_nolimiter(order):
    """
    advecting with a velocity of 1 should result in the mirror of
    advecting with a velocity of -1
    """
    forward_advection = AdvectionSolver(order=order, a=1)
    forward_advection.rkn(order)
    backward_advection = AdvectionSolver(order=order, a=-1)
    backward_advection.rkn(order)
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
    solution.rkn(order)
    assert [i > 0 and i < 1 for i in solution.u[-1]]
