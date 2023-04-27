# advection solution schems
import numpy as np
import matplotlib.pyplot as plt
from util.initial_condition import initial_condition2d
from util.integrate import Integrator
from util.fvscheme import ConservativeInterpolation


def rk4_Dt_adjust(h, L, order):
    """
    args:
        h:  cell size
        L:  1d domain size
        order:  accuracy requirement
    returns:
        Dt multiplication factor for rk4 to satisfy an order of accuracy
    """
    return (h / L) ** ((order - 4) / 4)


# class definition
class AdvectionSolver(Integrator):
    """
    args:
        u0_preset   string describing a pre-coded initial condition
        n:  tuple of number of cells in x and y
        x:  tuple of boundaries in x
        y:  tuple of boundaries in y
        T:  solving time
        a:  tuple of constant advection speed in x and y or callable function
        a_max:  tuple of abs max advection speeds if a is callable
        courant:    stability condition
        order:  accuracy requirement for polynomial interpolation
        bc_type:    string describing a pre-coded boudnary condition
        loglen: number of saved states
        adujst_time_step:   whether to reduce timestep for order >4
    returns:
        u   array of saved states
    """

    def __init__(
        self,
        u0_preset: str = "square",
        n: tuple = 32,
        x: tuple = (0, 1),
        y: tuple = None,
        T: float = 1,
        a: tuple = (1, 1),
        a_max: float = None,
        courant: float = 0.5,
        order: int = 1,
        bc_type: str = "periodic",
        loglen: int = 11,
        adujst_time_step: bool = False,
    ):
        # misc. attributes
        self.order = order
        self.adujst_time_step = adujst_time_step

        # spatial discretization
        self.n = (n, n) if isinstance(n, int) else n
        nx, ny = self.n
        y = x if y is None else y
        self.x_interface = np.linspace(
            x[0], x[1], num=nx + 1
        )  # cell interfaces
        self.x = 0.5 * (
            self.x_interface[:-1] + self.x_interface[1:]
        )  # cell centers
        self.Lx = x[1] - x[0]  # domain size in x
        self.hx = self.Lx / nx  # cell size in x
        self.y_interface = np.linspace(
            y[0], y[1], num=ny + 1
        )  # cell interfaces
        self.y = 0.5 * (
            self.y_interface[:-1] + self.y_interface[1:]
        )  # cell centers
        self.Ly = y[1] - y[0]  # domain size in y
        self.hy = self.Ly / ny  # cell size in y

        # constant advection velocity defined at cell interfaces
        if isinstance(a, tuple):
            self.a = a[0] * np.ones((ny, nx + 1))
            self.b = a[1] * np.ones((ny + 1, nx))
        if callable(a):
            self.a = a
        a_max = (1, 1) if a_max is None else a_max

        # time discretization
        self.courant = courant
        Dt = courant * min(
            self.hx / max(a_max[0], 1e-6), self.hy / max(a_max[1], 1e-6)
        )
        Dt_adjustment = 1
        if self.adujst_time_step and order > 4:
            Dt_adjustment_x = rk4_Dt_adjust(self.hx, self.Lx, order)
            Dt_adjustment_y = rk4_Dt_adjust(self.hy, self.Ly, order)
            Dt_adjustment = min(Dt_adjustment_x, Dt_adjustment_y)
            Dt = Dt * Dt_adjustment
            print(
                f"Decreasing timestep by a factor of {Dt_adjustment} to maintain",
                f" order {order} with rk4",
            )
        self.Dt_adjustment = Dt_adjustment
        n_timesteps = int(np.ceil(T / Dt))
        self.Dt = T / n_timesteps
        self.t = np.linspace(0, T, num=n_timesteps + 1)

        # initial/boundary conditions
        u0 = initial_condition2d(self.x, self.y, u0_preset)
        self.bc_type = bc_type

        # initialize integrator
        super().__init__(u0=u0, t=self.t, loglen=loglen)

        # interpolating values at cell interfaces
        left_interface_stensil = (
            ConservativeInterpolation.construct_from_order(order, "left")
            .nparray()
            .reshape(-1, 1, 1)
        )
        right_interface_stensil = (
            ConservativeInterpolation.construct_from_order(order, "right")
            .nparray()
            .reshape(-1, 1, 1)
        )
        self._stensil_size = len(
            left_interface_stensil
        )  # assume symmetric stensils
        self.k = int(
            np.floor(len(left_interface_stensil) / 2)
        )  # length of stensil arms
        self.gw = self.k + 1

        # complete list of interpolation stensils going fromleft to right
        interior_stensils = []
        self.list_of_stensils = (
            [left_interface_stensil]
            + interior_stensils
            + [right_interface_stensil]
        )

    def apply_bc(
        self, u_without_ghost_cells: np.ndarray, gw: int, direction: str
    ) -> np.ndarray:
        """
        args:
            u_without_ghost_cells   1d np array
            gw      number of ghost cells on either side of u
            direction   x or y
        returns:
            u_with_ghost_cells
        """
        if self.bc_type == "periodic":
            if direction == "x":
                pad_width = ((0, 0), (gw, gw))
            if direction == "y":
                pad_width = ((gw, gw), (0, 0))
            u_with_ghost_cells = np.pad(
                u_without_ghost_cells, pad_width, mode="wrap"
            )
        return u_with_ghost_cells

    def riemann(
        self,
        velocities_at_boundaries: float,
        left_of_boundary_values: float,
        right_of_boundary_values: float,
    ) -> float:
        fluxes = np.zeros(velocities_at_boundaries.shape)
        fluxes = np.where(
            velocities_at_boundaries > 0, left_of_boundary_values, fluxes
        )
        fluxes = np.where(
            velocities_at_boundaries < 0, right_of_boundary_values, fluxes
        )
        return fluxes * velocities_at_boundaries

    def udot(self, u: np.ndarray, t_i: float, dt: float = None) -> np.ndarray:
        # construct an extended array with ghost zones and apply bc
        u_extended_ns = self.apply_bc(u, self.gw, direction="y")
        u_extended_ew = self.apply_bc(u, self.gw, direction="x")
        ny, nx = u.shape
        # construct an array of staggered state arrays such that a 3D matrix operation
        # with a stensil multiplies weights by their referenced cells north/south
        # east/west
        ns_stack, ew_stack = [], []
        for i in range(self._stensil_size):
            top_ind = i + ny + 2
            right_ind = i + nx + 2
            ew_stack.append(u_extended_ew[:, i:right_ind])
            ns_stack.append(u_extended_ns[i:top_ind, :])
        ns_stack, ew_stack = np.asarray(ns_stack), np.asarray(ew_stack)
        # evaluate 1d average at each cell face
        ubar_north = np.sum(ns_stack * self.list_of_stensils[1], axis=0) / sum(
            self.list_of_stensils[1]
        )
        ubar_south = np.sum(ns_stack * self.list_of_stensils[0], axis=0) / sum(
            self.list_of_stensils[0]
        )
        ubar_east = np.sum(ew_stack * self.list_of_stensils[1], axis=0) / sum(
            self.list_of_stensils[1]
        )
        ubar_west = np.sum(ew_stack * self.list_of_stensils[0], axis=0) / sum(
            self.list_of_stensils[0]
        )
        # solve flux at each face
        ew_fluxes = self.riemann(self.a, ubar_east[:, :-1], ubar_west[:, 1:])
        ns_fluxes = self.riemann(self.b, ubar_north[:-1, :], ubar_south[1:, :])
        return (
            -(ew_fluxes[:, 1:] - ew_fluxes[:, :-1]) / self.hx
            + -(ns_fluxes[1:, :] - ns_fluxes[:-1, :]) / self.hy
        )

    def rkorder(self):
        """
        rk integrate to an order that matches the spatial order
        """
        if self.order > 3:
            self.rk4()
        elif self.order > 2:
            self.rk3()
        elif self.order > 1:
            self.rk2()
        else:
            self.euler()

    def find_error(self, norm: str = "l1"):
        """
        measure error between first state and last state
        """
        approx = self.u[-1]
        truth = self.u[0]
        if norm == "l1":
            return np.sum(np.abs(approx - truth)) / len(truth)
        if norm == "l2":
            return np.sqrt(np.sum(np.power(approx - truth, 2)) / len(truth))
        if norm == "inf":
            return np.max(np.abs(approx - truth))

    def plot_error(self):
        fig, ((ax0, ax1, ax2)) = plt.subplots(1, 3, figsize=(12, 10))
        im0 = ax0.imshow(
            self.u[0], extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]]
        )
        ax0.figure.colorbar(im0, ax=ax0, shrink=0.5)
        ax0.set_title("initial condition")
        im1 = ax1.imshow(
            self.u[-1], extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]]
        )
        ax1.figure.colorbar(im1, ax=ax1, shrink=0.5)
        ax1.set_title("final time step")
        im2 = ax2.imshow(
            self.u[-1] - self.u[0],
            extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]],
        )
        ax2.figure.colorbar(im2, ax=ax2, shrink=0.5)
        ax2.set_title("initial condition - final time step")
        fig.tight_layout()
        plt.show()
