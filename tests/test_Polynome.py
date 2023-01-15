# test Polynome class, which also tests the LinearCombination class
import pytest
from random import sample, randint, random
from util.polynome import Polynome


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
    assert (rand_poly.eval(x + h) - rand_poly.eval(x - h)) / (
        2 * h
    ) == pytest.approx(rand_poly.prime().eval(x), abs=1e-3)


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_derivative_of_a_sum(unused_parameter):
    """
    (f(x) + g(x))' = f'(x) + g'(x)
    """
    rand_poly1 = create_rand_poly()
    rand_poly2 = create_rand_poly()
    assert (
        rand_poly1 + rand_poly2
    ).prime() == rand_poly1.prime() + rand_poly2.prime()


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_derivative_of_a_product(unused_parameter):
    """
    (f(x) * g(x))' = f'(x) * g(x) + f(x) * g'(x)
    """
    rand_poly1 = create_rand_poly()
    rand_poly2 = create_rand_poly()
    derivative_of_products = (rand_poly1 * rand_poly2).prime()
    product_rule = (
        rand_poly1 * rand_poly2.prime() + rand_poly1.prime() * rand_poly2
    )
    assert derivative_of_products == product_rule


def test_legendre():
    """
    test generating the nth Legendre polynomial
    """
    assert Polynome.legendre(0) == Polynome({0: 1})
    assert Polynome.legendre(1) == Polynome({1: 1})
    assert Polynome.legendre(2) == (1 / 2) * Polynome({2: 3, 0: -1})
    assert Polynome.legendre(3) == (1 / 2) * Polynome({3: 5, 1: -3})
    assert Polynome.legendre(4) == (1 / 8) * Polynome({4: 35, 2: -30, 0: 3})


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_find_zeros_integer(unused_parameter):
    """
    test generating as list of zeros using integer zeros
    """
    # create a list of up to 5 zeros beween x = a and b
    a = -5
    b = 5
    n_zeros = randint(0, 5)
    zeros = []
    for _ in range(n_zeros):
        new_int = randint(a, b)
        while new_int in zeros:
            new_int = randint(a, b)
        zeros.append(new_int)
    # create a polynomial with these zeros
    poly = Polynome({0: 1})
    for z in zeros:
        poly *= Polynome({1: 1, 0: -z})  # (x - z)
    zeros.sort()  # sort the list of zeros

    assert poly.find_zeros([a - 1, b + 1]) == pytest.approx(
        zeros, rel=None, abs=1e-6
    )


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_find_zeros_float(unused_parameter):
    """
    test generating as list of zeros using float zeros
    """
    # create a list of up to 5 zeros beween x = a and b
    a = -5
    b = 5
    n_zeros = randint(0, 5)
    zeros = []
    for _ in range(n_zeros):
        new_int = randint(a, b) * random()
        while any(
            abs(existing_zero - new_int) < 0.1 for existing_zero in zeros
        ):
            new_int = randint(a, b) * random()
        zeros.append(new_int)
    # create a polynomial with these zeros
    poly = Polynome({0: 1})
    for z in zeros:
        poly *= Polynome({1: 1, 0: -z})  # (x - z)
    zeros.sort()  # sort the list of zeros

    assert poly.find_zeros([a - 1, b + 1]) == pytest.approx(
        zeros, rel=None, abs=1e-6
    )
