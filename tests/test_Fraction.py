# test the Fraction module
import pytest
from numpy.random import randint
from finite_volume.mathematiques import Fraction


n_tests = 10
max_int = 20


# helper functions
@pytest.fixture
def frac():
    """
    generate a random fraction
    """
    numerator = randint(-max_int - 1, max_int + 1)
    denominator = randint(-max_int - 1, max_int + 1)
    while denominator == 0:  # denominator can't be 0
        denominator = randint(-max_int - 1, max_int + 1)
    return Fraction(numerator, denominator)


# tests
@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_zero(unused_parameter, frac):
    """
    fraction with zero numerator
    """
    denom = randint(-max_int, max_int)
    assert Fraction(0, denom) == Fraction.zero()


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_addition_and_subtraction(unused_parameter, frac):
    """
    subtracting a fraction from one and then adding it back again should
    return one
    """
    fraction = frac
    assert (Fraction(1, 1) - fraction) + fraction == Fraction(1, 1)


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_zero_sum(unused_parameter, frac):
    """
    adding a fraction to zero should return that same fraction
    """
    fraction = frac
    assert Fraction.zero() + fraction == fraction


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_inverse_element(unused_parameter, frac):
    """
    dividing one by a fraction should return the inverse of that fraction
    """
    fraction = frac
    while fraction.numerator == 0:
        fraction.numerator = randint(-max_int - 1, max_int + 1)
    assert (Fraction(1, 1) / fraction) == Fraction(
        fraction.denominator, fraction.numerator
    )


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_abs(unused_parameter, frac):
    """
    absolute value of a fraction
    """
    fraction = frac
    absfrac = abs(fraction)
    if fraction.numerator < 0:
        assert absfrac.numerator == -fraction.numerator
        assert absfrac.denominator == fraction.denominator
    else:
        assert absfrac.numerator == fraction.numerator
        assert absfrac.denominator == fraction.denominator
