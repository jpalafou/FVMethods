import pytest
import numpy as np
import math
import random
from util.integrate import Integrator


n_tests = 5


@pytest.mark.parametrize("unused_parameter", range(n_tests))
def test_damped_oscillator(unused_parameter):
    """
    test rk4 integration with a damped oscillator
    x_ddot + b x_dot + (k / m) = 0 at t = T
    """
    T = 5
    m = 1
    b = random.randint(0, 2)
    k = random.randint(0, 5)
    x0 = random.randint(-1, 1)
    xdot0 = random.randint(-1, 1)
    while k == b**2 / (4 * m) or (k / m <= (b**2 / (4 * m**2))):
        k = random.randint(0, 5)
    t = np.arange(0, T + 0.001, 0.001)
    omega = np.sqrt((k / m) - (b**2 / (4 * m**2)))

    # find analytical solution
    phi0 = math.atan2(-(xdot0 + b * x0 / (2 * m)) / omega, x0)
    if np.cos(phi0) != pytest.approx(0):
        A = x0 / np.cos(phi0)
    else:
        A = -(b * x0 / (2 * m) + xdot0) / (omega * np.sin(phi0))

    def analytical_solution(t):
        return A * np.exp(-(b * t) / (2 * m)) * np.cos(omega * t + phi0)

    # find solution using rk4
    class DampedOscillatorTest(Integrator):
        """
        state vector: (x, xdot)
        """

        def xdot(self, x, t_i):
            return np.array([x[1], -b * x[1] - (k / m) * x[0]])

    dho = DampedOscillatorTest(np.array([x0, xdot0]), t)
    dho.rk4()

    # compare
    assert dho.x[0, -1] == pytest.approx(analytical_solution(t[-1]))
