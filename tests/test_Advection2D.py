import pytest
import numpy as np
from util.advection2d import AdvectionSolver


def test_meshsize_convergence():
    """
    l1 error should decrease with finer meshes
    """
    nlist = [16, 32, 64]
    errorlist = np.array([])
    for n in nlist:
        solution = AdvectionSolver(
            u0_preset="sinus",
            n=n,
            x=(0, 1),
            v=(1, 2),
            T=1,
            courant=0.5,
            order=3,
            apriori_limiting=None,
        )
        solution.rkorder()
        errorlist = np.append(errorlist, solution.find_error("l1"))
    assert all(errorlist[:-1] - errorlist[1:] > 0)

def test_order_convergence():
    """
    l1 error should decrease with increasing solution order
    """
    orderlist = [1, 2, 3, 4, 5]
    errorlist = np.array([])
    for order in orderlist:
        solution = AdvectionSolver(
            u0_preset="sinus",
            n=64,
            x=(0, 1),
            v=(1, 2),
            T=1,
            courant=0.5,
            order=order,
            apriori_limiting=None,
        )
        solution.rkorder()
        errorlist = np.append(errorlist, solution.find_error("l1"))
    assert all(errorlist[:-1] - errorlist[1:] > 0)