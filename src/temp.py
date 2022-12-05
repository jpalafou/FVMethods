import numpy as np
import math
from util.integrate import Integrator
import matplotlib.pyplot as plt


m = 1
b = 1
k = 1
x0 = 1
xdot0 = 1
t = np.arange(0, 3.01, 0.01)

omega = np.sqrt((k / m) - (b ** 2 / (4 * m ** 2)))

# find analytical solution
phi0 = math.atan2(-(xdot0 + b * x0 / (2 * m)) / omega, x0)
if np.cos(phi0) != 0:
    A = x0 / np.cos(phi0)
else:
    A = -(b * x0 / (2 * m) + xdot0) / (omega * np.sin(phi0))

def analytical_solution(t):
    return A * np.exp(-(b * t) / (2 * m)) * np.cos(omega * t + phi0)

class DampedOscillatorTest(Integrator):
    """
    state vector: (x, xdot)
    """
    def xdot(self, x, t_i):
        return np.array([x[1], -b * x[1] - (k / m) * x[0]])

dho = DampedOscillatorTest(np.array([x0, xdot0]), t)
dho.rk4()
# assert dho.x[0,-1] == pytest.approx(analytical_solution(t[-1]))
plt.plot(t, analytical_solution(t), t, dho.x[0, :])
plt.show()