import pytest
from util.lincom import Fraction
from util.fvscheme import Kernel, Interpolation


# ADD: randomized evaluation of x


def test_2nd_order_right_biased_both_faces():
    solution_right = {0: Fraction((1, 2)), 1: Fraction((1, 2))}
    assert Interpolation.construct(Kernel(0, 1), 'right').coeffs == solution_right

    solution_left = {0: Fraction((3, 2)), 1: Fraction((-1, 2))}
    assert Interpolation.construct(Kernel(0, 1), 'left').coeffs == solution_left


def test_2nd_order_left_biased_both_faces():
    solution_right = {-1: Fraction((-1, 2)), 0: Fraction((3, 2))}
    assert Interpolation.construct(Kernel(1, 0), 'right').coeffs == solution_right

    solution_left = {-1: Fraction((1, 2)), 0: Fraction((1, 2))}
    assert Interpolation.construct(Kernel(1, 0), 'left').coeffs == solution_left


def test_2nd_order_central_difference():
    Teyssier_solution = {-1: Fraction((-1, 2)), 1: Fraction((1, 2))}
    my_solution = Interpolation.construct(Kernel(0,1), 'r') - Interpolation.construct(Kernel(1,0), 'l')
    assert my_solution.coeffs == Teyssier_solution


def test_2nd_order_Fromm():
    right_solution = {-1: Fraction((-1, 4)), 0: Fraction((1, 1)), 1: Fraction((1, 4))}
    right_average = (Interpolation.construct(Kernel(0, 1), 'r') + Interpolation.construct(Kernel(1, 0), 'r')) / 2
    assert right_average.coeffs == right_solution

    left_solution = {-1: Fraction((1, 4)), 0: Fraction((1, 1)), 1: Fraction((-1, 4))}
    left_average = (Interpolation.construct(Kernel(0, 1), 'l') + Interpolation.construct(Kernel(1, 0), 'l')) / 2
    assert left_average.coeffs == left_solution

    du_solution = {-2: Fraction((1, 4)), -1: Fraction((-5, 4)), 0: Fraction((3, 4)), 1: Fraction((1, 4))}
    my_du = (Interpolation.construct(Kernel(0, 1), 'r') + Interpolation.construct(Kernel(1, 0), 'r')) / 2 - (Interpolation.construct(Kernel(0, 1, -1), 'r') + Interpolation.construct(Kernel(1, 0, -1), 'r')) / 2
    assert my_du.coeffs == du_solution


def test_3rd_order_both_faces():
    solution_right = {-1: Fraction((-1, 6)), 0: Fraction((5, 6)), 1: Fraction((1, 3))}
    assert Interpolation.construct(Kernel(1, 1), 'right').coeffs == solution_right

    solution_left = {-1: Fraction((1, 3)), 0: Fraction((5, 6)), 1: Fraction((-1, 6))}
    assert Interpolation.construct(Kernel(1, 1), 'left').coeffs == solution_left


def test_3rd_order_difference():
    du_solution = {-2: Fraction((1, 6)), -1: Fraction((-1, 1)), 0: Fraction((1, 2)), 1: Fraction((1, 3))}
    my_du = (Interpolation.construct(Kernel(1, 1), 'r') - Interpolation.construct(Kernel(1, 1, -1), 'r')).coeffs
    assert my_du == du_solution


def test_4th_order_right_biased_right_face():
    solution_right = {-1: Fraction((-1, 12)), 0: Fraction((7, 12)), 1: Fraction((7, 12)), 2: Fraction((-1, 12))}
    assert Interpolation.construct(Kernel(1, 2), 'right').coeffs == solution_right


def test_8th_order_right_biased_left_face():
    f = 840
    solution_left = {
        -3: Fraction((5, f)),
        -2: Fraction((-55, f)),
        -1: Fraction((365, f)),
        0: Fraction((743, f)),
        1: Fraction((-307, f)),
        2: Fraction((113, f)),
        3: Fraction((-27, f)),
        4: Fraction((3, f)),
    }
    assert Interpolation.construct(Kernel(3, 4), 'left').coeffs == solution_left
