import pytest
from src.fvscheme import Kernel


def test_2nd_order_right_biased():
    solution_right = {0: 1 / 2, 1: 1 / 2}
    assert Kernel(0, 1).solve_u_at_interface("r") == solution_right

    solution_left = {0: 3 / 2, 1: -1 / 2}
    assert Kernel(0, 1).solve_u_at_interface("l") == solution_left


def test_2nd_order_left_biased():
    solution_right = {-1: -1 / 2, 0: 3 / 2}
    assert Kernel(1, 0).solve_u_at_interface("r") == solution_right

    solution_left = {-1: 1 / 2, 0: 1 / 2}
    assert Kernel(1, 0).solve_u_at_interface("l") == solution_left


def test_3rd_order():
    solution_right = {-1: -1 / 6, 0: 5 / 6, 1: 2 / 6}
    assert Kernel(1, 1).solve_u_at_interface("r") == pytest.approx(solution_right)

    solution_left = {-1: 2 / 6, 0: 5 / 6, 1: -1 / 6}
    assert Kernel(1, 1).solve_u_at_interface("l") == pytest.approx(solution_left)


def test_4th_order_right_biased():
    solution_right = {-1: -1 / 12, 0: 7 / 12, 1: 7 / 12, 2: -1 / 12}
    assert Kernel(1, 2).solve_u_at_interface("r") == pytest.approx(solution_right)
    # only testing right interface


def test_8th_order_right_biased():
    # only testing left interface
    f = 840
    solution_left = {
        -3: 5 / f,
        -2: -55 / f,
        -1: 365 / f,
        0: 743 / f,
        1: -307 / f,
        2: 113 / f,
        3: -27 / f,
        4: 3 / f,
    }
    assert Kernel(3, 4).solve_u_at_interface("l") == pytest.approx(solution_left)
