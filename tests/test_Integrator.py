import pytest
import numpy as np
import math
import random
from finite_volume.integrate import Integrator


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
    u0 = random.randint(-1, 1)
    udot0 = random.randint(-1, 1)
    while k == b**2 / (4 * m) or (k / m <= (b**2 / (4 * m**2))):
        k = random.randint(0, 5)
    t = np.arange(0, T + 0.001, 0.001)
    omega = np.sqrt((k / m) - (b**2 / (4 * m**2)))

    # find analytical solution
    phi0 = math.atan2(-(udot0 + b * u0 / (2 * m)) / omega, u0)
    if np.cos(phi0) != pytest.approx(0):
        A = u0 / np.cos(phi0)
    else:
        A = -(b * u0 / (2 * m) + udot0) / (omega * np.sin(phi0))

    def analytical_solution(t):
        return A * np.exp(-(b * t) / (2 * m)) * np.cos(omega * t + phi0)

    # find solution using rk4
    class DampedOscillatorTest(Integrator):
        """
        state vector: (u, udot)
        """

        def udot(self, u, t_i, dt=None):
            return np.array([u[1], -b * u[1] - (k / m) * u[0]])

    dho = DampedOscillatorTest(np.array([u0, udot0]), t)
    dho.rk4()

    # compare
    assert dho.u[-1][0] == pytest.approx(analytical_solution(t[-1]))
