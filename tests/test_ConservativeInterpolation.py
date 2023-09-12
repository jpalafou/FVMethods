import pytest
import numpy as np
from random import random
from finite_volume.mathematiques import Fraction, Polynome
from finite_volume.fvscheme import Kernel, ConservativeInterpolation


n_tests = 5


def test_2nd_order_right_biased_both_faces():
    """
    2nd order right-biased scheme on both faces using construct_from_kernel
    method using Teyssier's solution
    """
    solution_right = {0: Fraction(1, 2), 1: Fraction(1, 2)}
    assert (
        ConservativeInterpolation.construct_from_kernel(Kernel(0, 1), "right").coeffs
        == solution_right
    )

    solution_left = {0: Fraction(3, 2), 1: Fraction(-1, 2)}
    assert (
        ConservativeInterpolation.construct_from_kernel(Kernel(0, 1), "left").coeffs
        == solution_left
    )


def test_2nd_order_left_biased_both_faces():
    """
    2nd order left-biased scheme on both faces using construct_from_kernel
    method using Teyssier's solution
    """
    solution_right = {-1: Fraction(-1, 2), 0: Fraction(3, 2)}
    assert (
        ConservativeInterpolation.construct_from_kernel(Kernel(1, 0), "right").coeffs
        == solution_right
    )

    solution_left = {-1: Fraction(1, 2), 0: Fraction(1, 2)}
    assert (
        ConservativeInterpolation.construct_from_kernel(Kernel(1, 0), "left").coeffs
        == solution_left
    )


def test_2nd_order_central_difference():
    """
    2nd order central difference scheme scheme using construct_from_kernel
    method using Teyssier's solution
    """
    Teyssier_solution = {-1: Fraction(-1, 2), 1: Fraction(1, 2)}
    my_solution = ConservativeInterpolation.construct_from_kernel(
        Kernel(0, 1), "right"
    ) - ConservativeInterpolation.construct_from_kernel(Kernel(1, 0), "left")
    assert my_solution.coeffs == Teyssier_solution


def test_2nd_order_Fromm():
    """
    2nd order Fromm scheme on both faces using construct_from_kernel
    method using Teyssier's solution
    """
    right_solution = {
        -1: Fraction(-1, 4),
        0: Fraction(1, 1),
        1: Fraction(1, 4),
    }
    right_average = (
        ConservativeInterpolation.construct_from_kernel(Kernel(0, 1), "right")
        + ConservativeInterpolation.construct_from_kernel(Kernel(1, 0), "right")
    ) / 2
    assert right_average.coeffs == right_solution

    left_solution = {-1: Fraction(1, 4), 0: Fraction(1, 1), 1: Fraction(-1, 4)}
    left_average = (
        ConservativeInterpolation.construct_from_kernel(Kernel(0, 1), "left")
        + ConservativeInterpolation.construct_from_kernel(Kernel(1, 0), "left")
    ) / 2
    assert left_average.coeffs == left_solution

    du_solution = {
        -2: Fraction(1, 4),
        -1: Fraction(-5, 4),
        0: Fraction(3, 4),
        1: Fraction(1, 4),
    }
    my_du = (
        ConservativeInterpolation.construct_from_kernel(Kernel(0, 1), "right")
        + ConservativeInterpolation.construct_from_kernel(Kernel(1, 0), "right")
    ) / 2 - (
        ConservativeInterpolation.construct_from_kernel(Kernel(0, 1, -1), "right")
        + ConservativeInterpolation.construct_from_kernel(Kernel(1, 0, -1), "right")
    ) / 2
    assert my_du.coeffs == du_solution


def test_3rd_order_both_faces():
    """
    3rd order scheme on both faces using construct_from_kernel method
    using Teyssier's solution
    """
    solution_right = {
        -1: Fraction(-1, 6),
        0: Fraction(5, 6),
        1: Fraction(1, 3),
    }
    assert (
        ConservativeInterpolation.construct_from_kernel(Kernel(1, 1), "right").coeffs
        == solution_right
    )

    solution_left = {-1: Fraction(1, 3), 0: Fraction(5, 6), 1: Fraction(-1, 6)}
    assert (
        ConservativeInterpolation.construct_from_kernel(Kernel(1, 1), "left").coeffs
        == solution_left
    )


def test_3rd_order_difference():
    """
    2nd order difference scheme on both faces using construct_from_kernel
    method using Teyssier's solution
    """
    du_solution = {
        -2: Fraction(1, 6),
        -1: Fraction(-1, 1),
        0: Fraction(1, 2),
        1: Fraction(1, 3),
    }
    my_du = (
        ConservativeInterpolation.construct_from_kernel(Kernel(1, 1), "right")
        - ConservativeInterpolation.construct_from_kernel(Kernel(1, 1, -1), "right")
    ).coeffs
    assert my_du == du_solution


def test_4th_order_right_biased_right_face():
    """
    4nd order right biased scheme on both faces using construct_from_kernel
    method using Teyssier's solution
    """
    solution_right = {
        -1: Fraction(-1, 12),
        0: Fraction(7, 12),
        1: Fraction(7, 12),
        2: Fraction(-1, 12),
    }
    assert (
        ConservativeInterpolation.construct_from_kernel(Kernel(1, 2), "right").coeffs
        == solution_right
    )


def test_8th_order_right_biased_left_face():
    """
    8th order right biased scheme on both faces using construct_from_kernel
    method using Teyssier's solution
    """
    f = 840
    solution_left = {
        -3: Fraction(5, f),
        -2: Fraction(-55, f),
        -1: Fraction(365, f),
        0: Fraction(743, f),
        1: Fraction(-307, f),
        2: Fraction(113, f),
        3: Fraction(-27, f),
        4: Fraction(3, f),
    }
    assert (
        ConservativeInterpolation.construct_from_kernel(Kernel(3, 4), "left").coeffs
        == solution_left
    )


def test_construct_from_order():
    """
    construct a 5th order reconstruction scheme
    """
    assert ConservativeInterpolation.construct_from_kernel(
        Kernel(2, 2), "right"
    ) == ConservativeInterpolation.construct_from_order(5, "r")


def test_nparray():
    """
    construct a 5th order reconstruction scheme and convert it to an array
    """
    scheme = ConservativeInterpolation.construct_from_order(5, "l")
    scheme_np = scheme.nparray()
    assert all(
        np.array([i.numerator / i.denominator for i in list(scheme.coeffs.values())])
        == scheme_np / sum(scheme_np)
    )


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_stencil_construction_for_floating_point_evalation(unused_parameter):
    # def test_float_evalation():
    """
    construct a stencil for a 3 cell kernel with a known solution
    verify that floating point evaluations hold
    """
    kernel = Kernel(1, 1)
    h = kernel.h
    # known solution
    # u(x) = p1(x) u_i-1 + p2(x) u_i + p3(x) u_i+1
    p1 = Polynome({2: 3, 1: -6, 0: -1}) / 48
    p2 = Polynome({2: -3, 0: 13}) / 24
    p3 = Polynome({2: 3, 1: 6, 0: -1}) / 48
    # fake data
    u_bar_max = 20
    u_bar = np.array([random() * u_bar_max for _ in range(3)])
    # 5 reconstruction points
    exes = [random() - 0.5 for _ in range(5)]
    # construct stencil
    stencils = [
        ConservativeInterpolation.construct_from_kernel(kernel, x).nparray()
        for x in exes
    ]
    stencil_evaluations = [u_bar @ stencil / sum(stencil) for stencil in stencils]
    true_evaluations = [
        h
        * u_bar
        @ np.array(
            [
                p1.eval(h * x, as_fraction=False),
                p2.eval(h * x, as_fraction=False),
                p3.eval(h * x, as_fraction=False),
            ]
        )
        for x in exes
    ]
    assert stencil_evaluations == pytest.approx(true_evaluations)
