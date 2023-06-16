import dataclasses
import numpy as np
import abc


@dataclasses.dataclass
class Integrator:
    """
    for a system with a state vector u and a state derivate udot = f(u),
    solve for u at every t given an initial state vector u0
    """

    def __init__(
        self,
        u0: np.ndarray,
        t: np.ndarray,
        loglen: int = 10,
        aposteriori: bool = False,
    ):
        self._t = t
        self.loglen = len(t) if loglen is None else loglen
        self._ilog = [int(i) for i in np.linspace(0, len(t) - 1, self.loglen)]
        assert len(self._ilog) == len(
            set(self._ilog)
        )  # there should be no duplicate values
        assert t[self._ilog[0]] == t[0]
        assert t[self._ilog[-1]] == t[-1]
        self.t = t[self._ilog]
        self.emptyu = np.zeros(u0.shape)
        u = np.asarray([u0] + [self.emptyu for _ in range(self.loglen - 1)])
        self.u = u
        self.u0_initial = u0
        self.u0 = u0
        self.u1 = self.emptyu

    # helper functions
    @abc.abstractmethod
    def udot(self, u: np.ndarray, t_i: float) -> np.ndarray:
        """
        the state derivate at a given value of time
        """
        pass

    def findDt(self, i) -> float:
        return self._t[i + 1] - self._t[i]

    def logupdate(self, i):
        """
        store data in u every time the time index is a log index
        """
        if i + 1 in self._ilog:
            self.u[self._ilog.index(i + 1)] = self.u1

    def pre_integrate(self):
        """
        any producedures that are to be executed before time integration
        returns true or false for whether or not to proceed
        """
        return True

    def post_integrate(self):
        """
        any producedures that are to be executed after time integration
        """
        return

    # integrators
    def one_euler_step(self):
        """
        1st order forward Euler integrator
        """
        if not self.pre_integrate():
            return
        dt = self.findDt(0)
        # one stage with posteriori check
        self.u1 = self.u0 + dt * self.udot(self.u0, self._t[0])
        self.post_integrate()

    def euler(self):
        """
        1st order forward Euler integrator
        """
        if not self.pre_integrate():
            return
        for i in range(len(self._t) - 1):
            dt = self.findDt(i)
            # one stage with posteriori check
            self.u1 = self.u0 + dt * self.udot(self.u0, self._t[i])
            # clean up
            self.logupdate(i)
            self.u0 = self.u1
        self.post_integrate()

    def rk2(self):
        """
        2nd order Runge-Kutta integrator
        """
        if not self.pre_integrate():
            return
        for i in range(len(self._t) - 1):
            dt = self.findDt(i)
            # first stage with posteriori check
            k0 = self.udot(self.u0, self._t[i])
            stage1 = self.u0 + dt * k0
            # second stage with posteriori check
            k1 = self.udot(stage1, self._t[i] + dt)
            self.u1 = self.u0 + (1 / 2) * (k0 + k1) * dt
            # clean up
            self.logupdate(i)
            self.u0 = self.u1
        self.post_integrate()

    def rk3(self):
        """
        3rd order Runge-Kutta integrator
        """
        if not self.pre_integrate():
            return
        for i in range(len(self._t) - 1):
            dt = self.findDt(i)
            # first stage with posteriori check
            k0 = self.udot(self.u0, self._t[i])
            stage1 = self.u0 + (1 / 3) * dt * k0
            # second stage with posteriori check
            k1 = self.udot(stage1, self._t[i] + (1 / 3) * dt)
            stage2 = self.u0 + (2 / 3) * dt * k1
            # third stage with posteriori check
            k2 = self.udot(stage2, self._t[i] + (2 / 3) * dt)
            self.u1 = self.u0 + (1 / 4) * (k0 + 3 * k2) * dt
            # clean up
            self.logupdate(i)
            self.u0 = self.u1
        self.post_integrate()

    def rk4(self):
        """
        4th order Runge-Kutta integrator
        """
        if not self.pre_integrate():
            return
        for i in range(len(self._t) - 1):
            dt = self.findDt(i)
            # first stage with posteriori check
            k1 = self.udot(self.u0, self._t[i], dt / 2)
            stage1 = self.u0 + dt * k1 / 2
            # second stage with posteriori check
            k2 = self.udot(stage1, self._t[i] + dt / 2, dt / 2)
            stage2 = self.u0 + dt * k2 / 2
            # third stage with posteriori check
            k3 = self.udot(stage2, self._t[i] + dt / 2, dt)
            stage3 = self.u0 + dt * k3
            # fourth stage with posteriori check
            k4 = self.udot(stage3, self._t[i] + dt, dt)
            self.u1 = self.u0 + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * dt
            # clean up
            self.logupdate(i)
            self.u0 = self.u1
        self.post_integrate()

    def ssprk2(self):
        """
        2nd order strong stability preserving Runge-Kutta integrator
        """
        if not self.pre_integrate():
            return
        for i in range(len(self._t) - 1):
            dt = self.findDt(i)
            x1 = self.u0
            x2 = x1 + dt * self.udot(x1, self._t[i], dt)
            self.u1 = (1 / 2) * x1 + (1 / 2) * (x2 + dt * self.udot(x2, self._t[i], dt))
            self.logupdate(i)
            self.u0 = self.u1
        self.post_integrate()

    def ssprk3(self):
        """
        3rd order strong stability preserving Runge-Kutta integrator
        """
        if not self.pre_integrate():
            return
        for i in range(len(self._t) - 1):
            dt = self.findDt(i)
            x1 = self.u0
            x2 = x1 + dt * self.udot(x1, self._t[i], dt)
            x3 = (3 / 4) * x1 + (1 / 4) * (x2 + dt * self.udot(x2, self._t[i], dt))
            self.u1 = (1 / 3) * x1 + (2 / 3) * (x3 + dt * self.udot(x3, self._t[i], dt))
            self.logupdate(i)
            self.u0 = self.u1
        self.post_integrate()
