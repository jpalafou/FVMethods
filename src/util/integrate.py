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
        self.t = t
        self._ilog = [int(i) for i in np.linspace(0, len(t) - 1, loglen)]
        assert len(self._ilog) == len(
            set(self._ilog)
        )  # there should be no duplicate values
        assert t[self._ilog[0]] == t[0]
        assert t[self._ilog[-1]] == t[-1]
        self.tlog = t[self._ilog]
        self.emptyu = np.zeros(u0.shape)
        u = np.asarray([u0] + [self.emptyu for _ in range(loglen - 1)])
        self.u = u
        self.u0_initial = u0
        self.u0 = u0
        self.u1 = self.emptyu
        self.aposteriori = aposteriori

    # helper functions
    @abc.abstractmethod
    def udot(self, u: np.ndarray, t_i: float) -> np.ndarray:
        """
        the state derivate at a given value of time
        """
        pass

    @abc.abstractmethod
    def posteriori_revision(
        self, u0: np.ndarray, ucandidate: np.ndarray
    ) -> np.ndarray:
        """
        perform a posteriori check on the solution
        """
        pass

    def revise_the_candidate(self, u0: np.ndarray, ucandidate: np.ndarray):
        """
        perform a posteriori check on the solution
        """
        if self.aposteriori:
            ucandidate = self.posteriori_revision(u0=u0, ucandidate=ucandidate)

    def findDt(self, i) -> float:
        return self.t[i + 1] - self.t[i]

    def logupdate(self, i):
        """
        store data in u every time the time index is a log index
        """
        if i + 1 in self._ilog:
            self.u[self._ilog.index(i + 1)] = self.u1

    # integrators
    def euler(self):
        """
        1st order forward Euler integrator
        """
        for i in range(len(self.t) - 1):
            dt = self.findDt(i)
            # one stage with posteriori check
            self.u1 = self.u0 + dt * self.udot(self.u0, self.t[i])
            self.revise_the_candidate(self.u0, self.u1)
            # clean up
            self.logupdate(i)
            self.u0 = self.u1

    def rk2(self):
        """
        2nd order Runge-Kutta integrator
        """
        for i in range(len(self.t) - 1):
            dt = self.findDt(i)
            # first stage with posteriori check
            k0 = self.udot(self.u0, self.t[i])
            stage1 = self.u0 + dt * k0
            self.revise_the_candidate(self.u0, stage1)
            # second stage with posteriori check
            k1 = self.udot(stage1, self.t[i] + dt)
            self.u1 = self.u0 + (1 / 2) * (k0 + k1) * dt
            self.revise_the_candidate(self.u0, self.u1)
            # clean up
            self.logupdate(i)
            self.u0 = self.u1

    def rk3(self):
        """
        3rd order Runge-Kutta integrator
        """
        for i in range(len(self.t) - 1):
            dt = self.findDt(i)
            # first stage with posteriori check
            k0 = self.udot(self.u0, self.t[i])
            stage1 = self.u0 + (1 / 3) * dt * k0
            self.revise_the_candidate(self.u0, stage1)
            # second stage with posteriori check
            k1 = self.udot(stage1, self.t[i] + (1 / 3) * dt)
            stage2 = self.u0 + (2 / 3) * dt * k1
            self.revise_the_candidate(self.u0, stage2)
            # third stage with posteriori check
            k2 = self.udot(stage2, self.t[i] + (2 / 3) * dt)
            self.u1 = self.u0 + (1 / 4) * (k0 + 3 * k2) * dt
            self.revise_the_candidate(self.u0, self.u1)
            # clean up
            self.logupdate(i)
            self.u0 = self.u1

    def rk4(self):
        """
        4th order Runge-Kutta integrator
        """
        for i in range(len(self.t) - 1):
            dt = self.findDt(i)
            # first stage with posteriori check
            k0 = self.udot(self.u0, self.t[i])
            stage1 = self.u0 + dt * k0 / 2
            self.revise_the_candidate(self.u0, stage1)
            # second stage with posteriori check
            k1 = self.udot(stage1, self.t[i] + dt / 2)
            stage2 = self.u0 + dt * k1 / 2
            self.revise_the_candidate(self.u0, stage2)
            # third stage with posteriori check
            k2 = self.udot(stage2, self.t[i] + dt / 2)
            stage3 = self.u0 + dt * k2
            self.revise_the_candidate(self.u0, stage3)
            # fourth stage with posteriori check
            k3 = self.udot(stage3, self.t[i] + dt)
            self.u1 = self.u0 + (1 / 6) * (k0 + 2 * k1 + 2 * k2 + k3) * dt
            self.revise_the_candidate(self.u0, self.u1)
            # clean up
            self.logupdate(i)
            self.u0 = self.u1

    def ssp_rk2(self):
        """
        2nd order strong stability preserving Runge-Kutta integrator
        """
        for i in range(len(self.t) - 1):
            dt = self.findDt(i)
            x0 = self.u0
            k0 = self.udot(x0, self.t[i])
            x1 = x0 + k0 * dt
            k1 = self.udot(x1, self.t[i])
            self.u1 = (1 / 2) * x0 + (1 / 2) * (x1 + k1 * dt)
            self.logupdate(i)
            self.u0 = self.u1

    def ssp_rk3(self):
        """
        3rd order strong stability preserving Runge-Kutta integrator
        """
        for i in range(len(self.t) - 1):
            dt = self.findDt(i)
            x0 = self.u0
            k0 = self.udot(x0, self.t[i])
            x1 = x0 + k0 * dt
            k1 = self.udot(x1, self.t[i])
            x2 = (3 / 4) * x0 + (1 / 4) * (x1 + k1 * dt)
            k2 = self.udot(x2, self.t[i])
            self.u1 = (1 / 3) * x0 + (2 / 3) * (x2 + k2 * dt)
            self.logupdate(i)
            self.u0 = self.u1


def rk4_Dt_adjust(h, L, spatial_order):
    """
    how much do we need to reduce Dt for rk4 to preserve a given order
    of accuracy
    """
    return (h / L) ** ((spatial_order - 4) / 4)
