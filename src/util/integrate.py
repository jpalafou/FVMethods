import dataclasses
import numpy as np
import abc

@dataclasses.dataclass
class Integrator:
    """
    for a system with a state vector x and a state derivate xdot = f(x),
    solve for x at every t given an initial state vector x0
    """

    def __init__(self, x0, t):
        self.x0 = x0
        self.t = t
        self.x = np.concatenate((np.array([x0]).T, np.zeros([len(x0), len(t) - 1])), axis = 1)

    @abc.abstractmethod
    def xdot(self, x, t_i):
        """
        the state derivate at a given value of time
        """
        pass

    def rk4(self):
        """
        4th order Runge-Kutta integrator
        """
        for i in range(len(self.t) - 1):
            dt = self.t[i + 1] - self.t[i]
            k0 = dt * self.xdot(self.x[:, i], self.t[i])
            k1 = dt * self.xdot(self.x[:, i] + k0 / 2, self.t[i])
            k2 = dt * self.xdot(self.x[:, i] + k1 / 2, self.t[i])
            k3 = dt * self.xdot(self.x[:, i] + k2, self.t[i])
            self.x[:, i + 1] = self.x[:, i] + (1 / 6) * (k0 + 2 * k1 + 2 * k2 + k3)