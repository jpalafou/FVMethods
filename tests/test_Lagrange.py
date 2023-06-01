# test Polynome class, which also tests the LinearCombination class
import pytest
from random import sample, randint
from finite_volume.mathbasic import Fraction
from finite_volume.fvscheme import Kernel
from finite_volume.polynome import Lagrange


n_tests = 5
l_max = 4
r_max = 4


# helper functions
def random_kernel():
    """
    generate a random kernel
    """
    return Kernel(randint(0, l_max), randint(0, r_max), randint(-l_max, r_max))


def random_Lagrange():
    """
    generate a random Lagrange polynomial
    """
    kern = random_kernel()
    return Lagrange.Lagrange_i(kern.x_cell_faces, randint(0, kern.size))


# tests
@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_Lagrange_init_Kernel_size(unused_parameter):
    """
    the degree of the polynomial should match the size of the kernel
    """
    kern = random_kernel()
    lagrange = Lagrange.Lagrange_i(kern.x_cell_faces, randint(0, kern.size))
    assert kern.size == max(lagrange.numerator.coeffs.keys())


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_Lagrange_init_cell_faces(unused_parameter):
    """
    the polynomial should return 0 when evaluated at cell faces
    """
    kern = random_kernel()
    x = kern.x_cell_faces
    i = randint(0, kern.size)
    lagrange = Lagrange.Lagrange_i(x, i)
    rand_x = sample([j for j in x if j != x[i]], 1)[0]
    # we shouldn't have sampled the ith x face value
    assert rand_x != kern.x_cell_faces[i]
    assert lagrange.numerator.eval(rand_x) == 0


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_zero_init(unused_parameter):
    """
    the zero method of an instance should return an instance with the same
    denominator and a zero polynomial as the numerator
    """
    lagrange = random_Lagrange()
    assert lagrange.denominator == lagrange.zero().denominator
    assert lagrange.zero().numerator == lagrange.numerator.__class__.zero()


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_sum(unused_parameter):
    """
    test finding a like denominator
    """
    lagrange1 = random_Lagrange()
    lagrange2 = random_Lagrange()
    x = randint(-l_max, r_max)
    assert (lagrange1 + lagrange2).eval(x) == lagrange1.eval(x) + lagrange2.eval(x)


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_zero_sum(unused_parameter):
    """
    adding zero to an instance should return the same instance
    """
    lagrange = random_Lagrange()
    assert lagrange + lagrange.zero() == lagrange


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_zero_diff(unused_parameter):
    """
    likewise for subtraction
    """
    lagrange = random_Lagrange()
    assert lagrange.zero() - lagrange == -lagrange


def test_eval_true():
    """
    (-x^3 + 8x^2 - 12x) / 16 has (1, -5 / 16) and (3, 9 / 16)
    test true division
    """
    polynome = Lagrange(coeffs={3: -1, 2: 8, 1: -12}, denominator=16)
    assert polynome.eval(1, "true") == -0.3125
    assert polynome.eval(3) == 0.5625


def test_eval_floor():
    """
    (-x^3 + 8x^2 - 12x) / 16 has (1, -5 / 16) and (3, 9 / 16)
    test floor division
    """
    polynome = Lagrange(coeffs={3: -1, 2: 8, 1: -12}, denominator=16)
    assert polynome.eval(1, "floor") == -1
    assert polynome.eval(3, "floor") == 0


def test_eval_fraction():
    """
    (-x^3 + 8x^2 - 12x) / 16 has (1, -5 / 16) and (3, 9 / 16)
    test fraction division
    """
    polynome = Lagrange(coeffs={3: -1, 2: 8, 1: -12}, denominator=16)
    assert polynome.eval(1, "fraction") == -Fraction(5, 16)
    assert polynome.eval(3, "fraction") == Fraction(9, 16)
