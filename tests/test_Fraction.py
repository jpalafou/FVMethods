# test the Fraction module
import pytest
import numpy as np
from finite_volume.mathematiques import Fraction


n_tests = 10
max_int = 20


# helper functions
@pytest.fixture
def frac():
    """
    generate a random fraction
    """
    numerator = np.random.randint(-max_int - 1, max_int + 1)
    denominator = np.random.randint(-max_int - 1, max_int + 1)
    while denominator == 0:  # denominator can't be 0
        denominator = np.random.randint(-max_int - 1, max_int + 1)
    return Fraction(numerator, denominator)


# tests
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
        fraction.numerator = np.random.randint(-max_int - 1, max_int + 1)
    assert (Fraction(1, 1) / fraction) == Fraction(
        fraction.denominator, fraction.numerator
    )
