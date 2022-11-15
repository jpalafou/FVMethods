import pytest  # run > pytest -s -q -rAd
import numpy as np
from util.lincom import Fraction

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
    while denominator == 0: # denominator can't be 0
        denominator = np.random.randint(-max_int - 1, max_int + 1)
    return Fraction((numerator, denominator))


# tests
@pytest.mark.parametrize("unused_parameter", range(n_tests))  # test n_tests times
def test_addition_and_subtraction(unused_parameter, frac):
    fraction = frac
    assert (Fraction((1, 1)) - fraction) + fraction == Fraction((1, 1))

@pytest.mark.parametrize("unused_parameter", range(n_tests))  # test n_tests times
def test_zero_sum(unused_parameter, frac):
    fraction = frac
    assert Fraction.zero() + fraction == fraction

@pytest.mark.parametrize("unused_parameter", range(n_tests))  # test n_tests times
def test_inverse_element(unused_parameter, frac):
    fraction = frac
    if fraction.fraction[0] == 0:
        fraction.fraction = (np.random.randint(-max_int - 1, max_int + 1), fraction.fraction[1])
    assert (Fraction((1, 1)) / fraction) == Fraction((fraction.fraction[1], fraction.fraction[0]))
