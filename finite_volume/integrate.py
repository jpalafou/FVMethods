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
        dt: float,
        snapshot_dt: float,
        num_snapshots: int = 1,
        dt_min: float = None,
        t0: float = 0.0,
        progress_bar: bool = False,
    ):
        """
        args:
            u0              np array, initial state
            dt              largest timestep
            snapshot_dt     dt for snapshots
            num_snapshots   number of times to evolve system by snapshot_dt
            dt_min          smallest timestep
            t0              starting time
            progress_bar    whether to print a progress bar in the loop
        """
        # initialize
        self.u0 = u0
        self.t0 = t0
        self.dt = dt
        self.snapshot_dt = snapshot_dt
        self.step_count = 0

        # check num_snapshots
        if num_snapshots < 0:
            raise BaseException("num_snapshots must be positive.")
        self.num_snapshots = num_snapshots

        # check dt_min
        if dt_min is not None:
            if dt_min < 0 or dt_min >= self.dt:
                raise BaseException(f"Invalid minimum timestep size {dt_min}")
        self.dt_min = dt_min

        # progress bar
        self.progress_bar = progress_bar
        if self.progress_bar:
            self.update_printout = self.update_progress_bar
        else:
            self.update_printout = lambda *args: None

    @abc.abstractmethod
    def udot(self, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        args:
            u   np array
            t   time at which u is defined
            dt  time to let u evolve
        returns:
            dudt evaluated at time t
        """
        pass

    def looks_good(self, u: np.ndarray) -> bool:
        """
        args:
            u   np array
        returns:
            bool    whether or not to proceed to next time
        """
        return True

    @abc.abstractmethod
    def snapshot(self):
        """
        datalogging at set time intervals
        """
        pass

    @abc.abstractmethod
    def step_cleanup(self):
        """
        runs after each update of self.t0
        """
        pass

    @abc.abstractmethod
    def refine_timestep(self, dt: float) -> float:
        """
        rule for adjusting timestep on the fly
        """
        pass

    @abc.abstractmethod
    def pre_integrate(self, method_name: str) -> bool:
        """
        any producedures that are to be executed before time integration
        args:
            method_name name of integration method
        returns:
            bool    whether or not to proceed
        """
        return True

    @abc.abstractmethod
    def post_integrate(self):
        """
        teardown procedures
        """
        pass

    def integrate(self, step, method_name: str, overwrite_snapshot_dt: float = None):
        """
        args:
            step                    function to calculate u1 from u0
            method_name             name of integrating step
            overwrite_snapshot_dt   does nothing if None
        overwrites:
            t, u
        """
        solving_time = self.num_snapshots * self.snapshot_dt  # not to be used in loop

        # check whether to procede to numerical integration
        if not self.pre_integrate(method_name=f"{method_name}_{solving_time}"):
            return

        # initialize progress bar
        progress_bar = None
        if self.progress_bar:
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"
            progress_bar = tqdm(total=solving_time, bar_format=bar_format)

        # time loop
        dt = self.dt  # initial timestep size
        snap_time = (
            self.snapshot_dt if overwrite_snapshot_dt is None else overwrite_snapshot_dt
        )
        snapshot_counter = 0
        starting_time = time.time()
        while snapshot_counter < self.num_snapshots:
            move_on = force_move_on = False
            self.step_count += 1
            while not move_on:
                u1 = step(u0=self.u0, t0=self.t0, dt=dt)
                if self.looks_good(u1) or force_move_on:
                    self.u0 = u1
                    self.t0 += dt
                    if self.t0 == snap_time:
                        self.snapshot()
                        snapshot_counter += 1
                        snap_time += self.snapshot_dt
                    move_on = True  # procees to next step
                    dt = self.dt  # reset dt
                    self.step_cleanup()
                    self.update_printout(progress_bar)
                else:
                    dt = self.refine_timestep(dt)
                # set timestep to dt_min if it is smaller
                if self.dt_min is not None and dt <= self.dt_min:
                    dt = self.dt_min
                    force_move_on = True
                # reduce timestep if the next timestep will bring us beyond snap_time
                if self.t0 + dt > snap_time:
                    dt = snap_time - self.t0
                    if self.dt_min is not None and dt < self.dt_min:
                        force_move_on = True
                if dt < 0:
                    raise BaseException("Negative dt encountered.")
        ellapsed_time = time.time() - starting_time
        if self.progress_bar:
            progress_bar.close()
        print()
        self.solution_time = ellapsed_time
        self.post_integrate()

    def update_progress_bar(self, progress_bar):
        progress_bar.n = self.t0
        progress_bar.refresh()

    # integrators
    def one_euler_step(self, n: int = 1):
        """
        1st order ODE integrator
        """

        def step(u0, t0, dt):
            k1 = self.udot(u=u0, t=t0, dt=dt)
            u1 = u0 + dt * k1
            return u1

        overwrite_snapshot_dt = 0.0
        for _ in range(n):
            overwrite_snapshot_dt += self.snapshot_dt

        self.integrate(
            step=step,
            method_name=inspect.currentframe().f_code.co_name,
            overwrite_snapshot_dt=overwrite_snapshot_dt,
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

    def rk6(self):
        """
        6th order Runge-Kutta integrator
        """

        def step(u0, t0, dt):
            v = np.sqrt(21)
            k1 = dt * self.udot(u=u0, t=t0, dt=dt)
            k2 = dt * self.udot(u=u0 + k1, t=t0 + dt, dt=dt)
            k3 = dt * self.udot(u=u0 + (3 * k1 + k2) / 8, t=t0 + dt / 2, dt=dt / 2)
            k4 = dt * self.udot(
                u=u0 + (8 * k1 + 2 * k2 + 8 * k3) / 27,
                t=t0 + (2 / 3) * dt,
                dt=(2 / 3) * dt,
            )
            k5 = dt * self.udot(
                u=u0
                + (
                    3 * (3 * v - 7) * k1
                    - 8 * (7 - v) * k2
                    + 48 * (7 - v) * k3
                    - 3 * (21 - v) * k4
                )
                / 392,
                t=t0 + (7 - v) * dt / 14,
                dt=(7 - v) * dt / 14,
            )
            k6 = dt * self.udot(
                u=u0
                + (
                    -5 * (231 + 51 * v) * k1
                    - 40 * (7 + v) * k2
                    - 320 * v * k3
                    + 3 * (21 + 121 * v) * k4
                    + 392 * (6 + v) * k5
                )
                / 1960,
                t=t0 + (7 + v) * dt / 14,
                dt=(7 + v) * dt / 14,
            )
            k7 = dt * self.udot(
                u=u0
                + (
                    15 * (22 + 7 * v) * k1
                    + 120 * k2
                    + 40 * (7 * v - 5) * k3
                    - 63 * (3 * v - 2) * k4
                    - 14 * (49 + 9 * v) * k5
                    + 70 * (7 - v) * k6
                )
                / 180,
                t=t0 + dt,
                dt=dt,
            )
            u1 = u0 + (9 * k1 + 64 * k3 + 49 * k5 + 49 * k6 + 9 * k7) / 180
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
            k1 = self.udot(u=u0, t=t0, dt=dt)
            k2 = self.udot(u=u0 + dt * k1, t=t0 + dt, dt=dt)
            k3 = self.udot(
                u=u0 + 0.25 * dt * k1 + 0.25 * dt * k2, t=t0 + 0.5 * dt, dt=0.5 * dt
            )
            u1 = u0 + (dt / 6) * (k1 + k2 + 4 * k3)
            return u1

        self.integrate(step=step, method_name=inspect.currentframe().f_code.co_name)
