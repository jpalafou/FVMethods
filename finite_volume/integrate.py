import dataclasses
import numpy as np
import abc
from tqdm import tqdm
import time
import inspect


@dataclasses.dataclass
class Integrator:
    """
    for a system with a state vector u and a state derivate udot = f(u),
    solve for u at every t given an initial state vector u0
    """

    def __init__(
        self,
        u0: np.ndarray,
        T: float,
        dt: float,
        dt_min: float = None,
        t0: float = 0.0,
        log_every: int = 10,
    ):
        """
        args:
            u0          np array, initial state
            T           final time
            dt          largest timestep
            t0          starting time
            log_every   number of iterations to store in array
        """
        self.log_every = log_every
        self.iteration_count = 0
        self.t0 = t0  # time of state u0
        self.T = T
        self.dt = dt
        self.dt_min = self.dt / 2**10 if dt_min is None else dt_min
        self.u0 = u0  # state entering iteration step
        self.u = u0[np.newaxis]  # array of state arrays
        self.t = np.array([t0])  # array of times corresponding to self.u
        self.loglen = 1  # number of logged states

    # helper functions
    @abc.abstractmethod
    def udot(self, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        args:
            u   np array
            t   time at which u is defined
            dt  time to let u evolve
        returns:
            dudt evaluated at time t    np array
        """
        pass

    def logupdate(self):
        """
        self.t0 has been overwritten with t0 + dt
        self.u0 has been overwritten with the state at t0 + dt
        """
        if self.iteration_count % self.log_every == 0 or self.t0 == self.T:
            self.t = np.append(self.t, self.t0)
            self.u = np.append(self.u, self.u0[np.newaxis], axis=0)
            self.loglen += 1

    def pre_integrate(self, method_name):
        """
        any producedures that are to be executed before time integration
        args:
            method_name name of integration method
        returns:
            bool    whether or not to proceed
        """
        return True

    def post_integrate(self):
        """
        any producedures that are to be executed after time integration
        """
        return

    def looks_good(self, u):
        """
        args:
            u   np array
        returns:
            bool    whether or not to proceed to next time
        """
        return True

    def integrate(self, step, method_name: str, T: float = None):
        """
        args:
            step    function to calculate u1 from u0
        overwrites:
            t, u
        """
        # time step
        solving_time = self.T if T is None else T

        # check whether to procede to numerical integration
        if not self.pre_integrate(method_name=f"{method_name}_{solving_time}"):
            return

        # initialize progress bar
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"
        progress_bar = tqdm(total=solving_time, bar_format=bar_format)

        # time loop
        dt = self.dt  # initial time step
        starting_time = time.time()
        while self.t0 < solving_time:
            move_on = False
            self.iteration_count += 1
            while not move_on:
                u1 = step(u0=self.u0, t0=self.t0, dt=dt)
                if self.looks_good(u1):
                    move_on = True
                    self.u0 = u1
                    self.t0 += dt
                    self.logupdate()
                    progress_bar.n = self.t0
                    progress_bar.refresh()
                    # reset dt
                    dt = self.dt
                    # reduce timestep if the next timestep will bring us beyond T
                    if self.t0 + dt > solving_time:
                        dt = solving_time - self.t0
                elif dt / 2 >= self.dt_min:
                    dt = dt / 2
        ellapsed_time = time.time() - starting_time
        self.solution_time = ellapsed_time
        progress_bar.close()
        print()
        self.post_integrate()

    # integrators
    def one_euler_step(self):
        """
        1st order ODE integrator
        """

        def step(u0, t0, dt):
            k1 = self.udot(u=u0, t=t0, dt=dt)
            u1 = u0 + dt * k1
            return u1

        self.integrate(
            step=step, method_name=inspect.currentframe().f_code.co_name, T=self.dt
        )

    def euler(self):
        """
        1st order ODE integrator
        """

        def step(u0, t0, dt):
            k1 = self.udot(u=u0, t=t0, dt=dt)
            u1 = u0 + dt * k1
            return u1

        self.integrate(step=step, method_name=inspect.currentframe().f_code.co_name)

    def rk2(self):
        """
        2nd order Runge-Kutta integrator
        """

        def step(u0, t0, dt):
            k1 = self.udot(u=u0, t=t0, dt=dt)
            k2 = self.udot(u=u0 + (1 / 2) * dt * k1, t=t0 + (1 / 2) * dt, dt=dt)
            u1 = u0 + dt * k2
            return u1

        self.integrate(step=step, method_name=inspect.currentframe().f_code.co_name)

    def rk3(self):
        """
        3th order Runge-Kutta integrator
        """

        def step(u0, t0, dt):
            k1 = self.udot(u=u0, t=t0, dt=dt)
            k2 = self.udot(
                u=u0 + (1 / 3) * dt * k1, t=t0 + (1 / 3) * dt, dt=(2 / 3) * dt
            )
            k3 = self.udot(
                u=u0 + (2 / 3) * dt * k2, t=t0 + (2 / 3) * dt, dt=(3 / 4) * dt
            )
            u1 = u0 + (1 / 4) * dt * (k1 + 3 * k3)
            return u1

        self.integrate(step=step, method_name=inspect.currentframe().f_code.co_name)

    def rk4(self):
        """
        4th order Runge-Kutta integrator
        """

        def step(u0, t0, dt):
            k1 = self.udot(u=u0, t=t0, dt=dt)
            k2 = self.udot(
                u=u0 + (1 / 2) * dt * k1, t=t0 + (1 / 2) * dt, dt=(1 / 2) * dt
            )
            k3 = self.udot(u=u0 + (1 / 2) * dt * k2, t=t0 + (1 / 2) * dt, dt=dt)
            k4 = self.udot(u=u0 + dt * k3, t=t0 + dt, dt=(1 / 6) * dt)
            u1 = u0 + (1 / 6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
            return u1

        self.integrate(step=step, method_name=inspect.currentframe().f_code.co_name)

    def ssprk2(self):
        """
        2nd order strong stability preserving Runge-Kutta integrator
        """

        def step(u0, t0, dt):
            _u1 = u0
            _u2 = _u1 + dt * self.udot(u=_u1, t=t0, dt=dt)
            u1 = (1 / 2) * _u1 + (1 / 2) * (_u2 + dt * self.udot(u=_u2, t=t0, dt=dt))
            return u1

        self.integrate(step=step, method_name=inspect.currentframe().f_code.co_name)

    def ssprk3(self):
        """
        3rd order strong stability preserving Runge-Kutta integrator
        """

        def step(u0, t0, dt):
            _u1 = u0
            _u2 = _u1 + dt * self.udot(u=_u1, t=t0, dt=dt)
            _u3 = (3 / 4) * _u1 + (1 / 4) * (_u2 + dt * self.udot(u=_u2, t=t0, dt=dt))
            u1 = (1 / 3) * _u1 + (2 / 3) * (_u3 + dt * self.udot(u=_u3, t=t0, dt=dt))
            return u1

        self.integrate(step=step, method_name=inspect.currentframe().f_code.co_name)
