import dataclasses
import numpy as np
import abc


@dataclasses.dataclass
class Integrator:
    """
    for a system with a state vector u and a state derivate udot = f(u),
    solve for u at every t given an initial state vector u0
    """

    def __init__(self, u0: np.ndarray, t: np.ndarray):
        self.u0 = u0
        self.t = t
        u = np.asarray([np.zeros(u0.shape) for _ in range(len(t))])
        u[0] = u0
        self.u = u

    @abc.abstractmethod
    def udot(self, u: np.ndarray, t_i: float) -> np.ndarray:
        """
        the state derivate at a given value of time
        """
        pass

    def euler(self):
        """
        1st order forward Euler integrator
        """
        for i in range(len(self.t) - 1):
            dt = self.t[i + 1] - self.t[i]
            self.u[i + 1] = self.u[i] + dt * self.udot(self.u[i], self.t[i])

    def rk2(self):
        """
        2nd order Runge-Kutta integrator
        """
        for i in range(len(self.t) - 1):
            dt = self.t[i + 1] - self.t[i]
            k0 = self.udot(self.u[i], self.t[i])
            k1 = self.udot(self.u[i] + dt * k0, self.t[i] + dt)
            self.u[i + 1] = self.u[i] + (1 / 2) * (k0 + k1) * dt

    def rk3(self):
        """
        3rd order Runge-Kutta integrator
        """
        for i in range(len(self.t) - 1):
            dt = self.t[i + 1] - self.t[i]
            k0 = self.udot(self.u[i], self.t[i])
            k1 = self.udot(
                self.u[i] + (1 / 3) * dt * k0, self.t[i] + (1 / 3) * dt
            )
            k2 = self.udot(
                self.u[i] + (2 / 3) * dt * k1, self.t[i] + (2 / 3) * dt
            )
            self.u[i + 1] = self.u[i] + (1 / 4) * (k0 + 3 * k2) * dt

    def rk4(self):
        """
        4th order Runge-Kutta integrator
        """
        for i in range(len(self.t) - 1):
            dt = self.t[i + 1] - self.t[i]
            k0 = self.udot(self.u[i], self.t[i])
            k1 = self.udot(self.u[i] + dt * k0 / 2, self.t[i] + dt / 2)
            k2 = self.udot(self.u[i] + dt * k1 / 2, self.t[i] + dt / 2)
            k3 = self.udot(self.u[i] + dt * k2, self.t[i] + dt)
            self.u[i + 1] = (
                self.u[i] + (1 / 6) * (k0 + 2 * k1 + 2 * k2 + k3) * dt
            )

    def rkn(self, n):
        """
        euler and rk2-4 integration
        """
        if not isinstance(n, int) or n < 1 or n > 4:
            raise BaseException(
                f"runge-kutta integration of order {n} not supported"
            )
        if n == 1:
            self.euler()
        elif n == 2:
            self.rk2()
        elif n == 3:
            self.rk3()
        elif n == 4:
            self.rk4()

    def ssp_rk2(self):
        """
        2nd order strong stability preserving Runge-Kutta integrator
        """
        for i in range(len(self.t) - 1):
            dt = self.t[i + 1] - self.t[i]
            x0 = self.u[i]
            k0 = self.udot(x0, self.t[i])
            x1 = x0 + k0 * dt
            k1 = self.udot(x1, self.t[i])
            self.u[i + 1] = (1 / 2) * x0 + (1 / 2) * (x1 + k1 * dt)

    def ssp_rk3(self):
        """
        3rd order strong stability preserving Runge-Kutta integrator
        """
        for i in range(len(self.t) - 1):
            dt = self.t[i + 1] - self.t[i]
            x0 = self.u[i]
            k0 = self.udot(x0, self.t[i])
            x1 = x0 + k0 * dt
            k1 = self.udot(x1, self.t[i])
            x2 = (3 / 4) * x0 + (1 / 4) * (x1 + k1 * dt)
            k2 = self.udot(x2, self.t[i])
            self.u[i + 1] = (1 / 3) * x0 + (2 / 3) * (x2 + k2 * dt)
