import numpy as np
from util.mathbasic import Fraction
from util.fvscheme import Kernel, PolynomialReconstruction


n_tests = 5


def test_2nd_order_right_biased_both_faces():
    """
    2nd order right-biased scheme on both faces using construct_from_kernel
    method using Teyssier's solution
    """
    solution_right = {0: Fraction(1, 2), 1: Fraction(1, 2)}
    assert (
        PolynomialReconstruction.construct_from_kernel(
            Kernel(0, 1), "right"
        ).coeffs
        == solution_right
    )

    solution_left = {0: Fraction(3, 2), 1: Fraction(-1, 2)}
    assert (
        PolynomialReconstruction.construct_from_kernel(
            Kernel(0, 1), "left"
        ).coeffs
        == solution_left
    )


def test_2nd_order_left_biased_both_faces():
    """
    2nd order left-biased scheme on both faces using construct_from_kernel
    method using Teyssier's solution
    """
    solution_right = {-1: Fraction(-1, 2), 0: Fraction(3, 2)}
    assert (
        PolynomialReconstruction.construct_from_kernel(
            Kernel(1, 0), "right"
        ).coeffs
        == solution_right
    )

    solution_left = {-1: Fraction(1, 2), 0: Fraction(1, 2)}
    assert (
        PolynomialReconstruction.construct_from_kernel(
            Kernel(1, 0), "left"
        ).coeffs
        == solution_left
    )


def test_2nd_order_central_difference():
    """
    2nd order central difference scheme scheme using construct_from_kernel
    method using Teyssier's solution
    """
    Teyssier_solution = {-1: Fraction(-1, 2), 1: Fraction(1, 2)}
    my_solution = PolynomialReconstruction.construct_from_kernel(
        Kernel(0, 1), "r"
    ) - PolynomialReconstruction.construct_from_kernel(Kernel(1, 0), "l")
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
        PolynomialReconstruction.construct_from_kernel(Kernel(0, 1), "r")
        + PolynomialReconstruction.construct_from_kernel(Kernel(1, 0), "r")
    ) / 2
    assert right_average.coeffs == right_solution

    left_solution = {-1: Fraction(1, 4), 0: Fraction(1, 1), 1: Fraction(-1, 4)}
    left_average = (
        PolynomialReconstruction.construct_from_kernel(Kernel(0, 1), "l")
        + PolynomialReconstruction.construct_from_kernel(Kernel(1, 0), "l")
    ) / 2
    assert left_average.coeffs == left_solution

    du_solution = {
        -2: Fraction(1, 4),
        -1: Fraction(-5, 4),
        0: Fraction(3, 4),
        1: Fraction(1, 4),
    }
    my_du = (
        PolynomialReconstruction.construct_from_kernel(Kernel(0, 1), "r")
        + PolynomialReconstruction.construct_from_kernel(Kernel(1, 0), "r")
    ) / 2 - (
        PolynomialReconstruction.construct_from_kernel(Kernel(0, 1, -1), "r")
        + PolynomialReconstruction.construct_from_kernel(Kernel(1, 0, -1), "r")
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
        PolynomialReconstruction.construct_from_kernel(
            Kernel(1, 1), "right"
        ).coeffs
        == solution_right
    )

    solution_left = {-1: Fraction(1, 3), 0: Fraction(5, 6), 1: Fraction(-1, 6)}
    assert (
        PolynomialReconstruction.construct_from_kernel(
            Kernel(1, 1), "left"
        ).coeffs
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
        PolynomialReconstruction.construct_from_kernel(Kernel(1, 1), "r")
        - PolynomialReconstruction.construct_from_kernel(Kernel(1, 1, -1), "r")
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
        PolynomialReconstruction.construct_from_kernel(
            Kernel(1, 2), "right"
        ).coeffs
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
        PolynomialReconstruction.construct_from_kernel(
            Kernel(3, 4), "left"
        ).coeffs
        == solution_left
    )


def test_construct_from_order():
    """
    construct a 5th order reconstruction scheme
    """
    assert PolynomialReconstruction.construct_from_kernel(
        Kernel(2, 2), "r"
    ) == PolynomialReconstruction.construct_from_order(5, "r")


def test_nparray():
    """
    construct a 5th order reconstruction scheme and convert it to an array
    """
    scheme = PolynomialReconstruction.construct_from_order(5, "l")
    scheme_np = scheme.nparray()
    assert all(
        np.array(
            [i.numerator / i.denominator for i in list(scheme.coeffs.values())]
        )
        == scheme_np / sum(scheme_np)
    )
