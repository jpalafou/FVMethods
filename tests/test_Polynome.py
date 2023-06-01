# test Polynome class, which also tests the LinearCombination class
import pytest
from random import sample, randint, random
from finite_volume.polynome import Polynome


n_tests = 10
max_degree = 10
max_coeff = 10


# helper functions
def create_rand_poly():
    """
    generate a random polynomial
    """
    degrees = sample(range(max_degree + 1), randint(0, max_degree + 1))
    coeffs = dict([(i, randint(-max_coeff, max_coeff)) for i in degrees])
    return Polynome(coeffs)


# tests
def test_zero_init():
    """
    creating a 0 polynomial
    """
    assert Polynome({}) == Polynome({0: 0})
    assert Polynome({}) == Polynome.zero()


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_zero_sum(unused_parameter):
    """
    adding zero to a polynomial should return that same polynomial
    """
    rand_poly = create_rand_poly()
    zero = Polynome.zero()
    assert rand_poly + zero == rand_poly


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_zero_diff(unused_parameter):
    """
    subtracting a polynomial from 0 should return the negative of that
    polynomial
    """
    rand_poly = create_rand_poly()
    zero = Polynome.zero()
    assert zero - rand_poly == -rand_poly


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_zero_product(unused_parameter):
    """
    multiplying a polynomial by 0 should return 0
    """
    rand_poly = create_rand_poly()
    zero = Polynome.zero()
    assert rand_poly * zero == zero


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_origin_solution(unused_parameter):
    """
    any polynomial with all terms having degree >= 1 should evaluate as 0 at
    x = 0
    """
    rand_poly = create_rand_poly()
    rand_poly.coeffs[0] = 0
    assert rand_poly.eval(0) == 0


def test_known_solution():
    """
    some arbiraty polynomial with a known solution at x = -5
    """
    poly = Polynome({5: 1 / 1000, 3: -1 / 1000, 1: 1 / 1000, 0: -3})
    assert poly.eval(-5) == -6.005


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_sum(unused_parameter):
    """
    h(x) = f(x) + g(x) for all x
    """
    rand_poly1 = create_rand_poly()
    rand_poly2 = create_rand_poly()
    x = 3 * (random() - 0.5)
    assert (rand_poly1 + rand_poly2).eval(x) == pytest.approx(
        rand_poly1.eval(x) + rand_poly2.eval(x)
    )


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_difference(unused_parameter):
    """
    h(x) = f(x) - g(x) for all x
    """
    rand_poly1 = create_rand_poly()
    rand_poly2 = create_rand_poly()
    x = 3 * (random() - 0.5)
    assert (rand_poly1 - rand_poly2).eval(x) == pytest.approx(
        rand_poly1.eval(x) - rand_poly2.eval(x)
    )


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_product(unused_parameter):
    """
    h(x) = f(x) * g(x) for all x
    """
    rand_poly1 = create_rand_poly()
    rand_poly2 = create_rand_poly()
    x = 3 * (random() - 0.5)
    assert (rand_poly1 * rand_poly2).eval(x) == pytest.approx(
        rand_poly1.eval(x) * rand_poly2.eval(x)
    )


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_derivative(unused_parameter):  # FIX large errors for small \abs{x}
    """
    finite difference approximation of a first derivative used to test the
    derivative of a polynimial
    """
    rand_poly = create_rand_poly()
    h = 0.000001
    x = sample([-3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3], 1)[0]
    assert (rand_poly.eval(x + h) - rand_poly.eval(x - h)) / (2 * h) == pytest.approx(
        rand_poly.prime().eval(x), abs=1e-3
    )


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_derivative_of_a_sum(unused_parameter):
    """
    (f(x) + g(x))' = f'(x) + g'(x)
    """
    rand_poly1 = create_rand_poly()
    rand_poly2 = create_rand_poly()
    assert (rand_poly1 + rand_poly2).prime() == rand_poly1.prime() + rand_poly2.prime()


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_derivative_of_a_product(unused_parameter):
    """
    (f(x) * g(x))' = f'(x) * g(x) + f(x) * g'(x)
    """
    rand_poly1 = create_rand_poly()
    rand_poly2 = create_rand_poly()
    derivative_of_products = (rand_poly1 * rand_poly2).prime()
    product_rule = rand_poly1 * rand_poly2.prime() + rand_poly1.prime() * rand_poly2
    assert derivative_of_products == product_rule
