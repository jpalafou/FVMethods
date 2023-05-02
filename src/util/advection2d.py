# advection solution schems
import numpy as np
from matplotlib import gridspec
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


def stack3d(u: np.ndarray, s: int, direction: str):
    """
    args:
        u   2d np array (m, n)
        s   number of arrays to stack
        direction   'x' or 'y'
    returns:
        3d array of stacked windows from u
        direction = 'x'
            (s, m, n - s + 1)
        direction = 'y'
            (s, m - s + 1, n)
    """
    ny, nx = u.shape
    list_of_smaller_arrays = []
    for i in range(s):
        if direction == "x":
            upper_bound = nx - s + i + 1
            list_of_smaller_arrays.append(u[:, i:upper_bound])
        elif direction == "y":
            upper_bound = ny - s + i + 1
            list_of_smaller_arrays.append(u[i:upper_bound, :])
    return np.asarray(list_of_smaller_arrays)


def apply_stensil(u: np.ndarray, stensil: np.ndarray):
    """
    args:
        u       3d np array (s, m, n)
        stensil 1d np array
    returns:
        linear combination of u along first dimension
        2d np array (m, n)
    """
    stensilcopy = stensil if stensil.ndim == 3 else stensil.reshape(-1, 1, 1)
    return np.sum(u * stensilcopy, axis=0) / sum(stensilcopy)


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
        n: tuple = (32, 32),
        x: tuple = (0, 1),
        y: tuple = None,
        T: float = 1,
        a: tuple = (1, 1),
        a_max: float = None,
        courant: float = 0.5,
        order: int = 1,
        bc_type: str = "periodic",
        loglen: int = 21,
        adujst_time_step: bool = False,
    ):
        # misc. attributes
        self.order = order
        self.adujst_time_step = adujst_time_step

        # spatial discretization
        self.n = (n, n) if isinstance(n, int) else n
        ny, nx = self.n
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
        self.h = self.hx if self.hx == self.hy else None

        # constant advection velocity defined at cell interfaces
        if isinstance(a, tuple):
            self.a = a[0] * np.ones((ny, nx + 1))
            self.b = a[1] * np.ones((ny + 1, nx))
        if callable(a):
            self.a = a
        a_max = (1, 1) if a_max is None else a_max

        # time discretization
        self.courant = courant
        if isinstance(a, tuple):
            Dt = courant / (
                np.max(np.abs(self.a)) / self.hx
                + np.max(np.abs(self.b)) / self.hy
            )
        if callable(a):
            precompa, precompb = self.a(self.x_interface, self.y_interface)
            maxa, maxb = np.max(np.abs(precompa)), np.max(np.abs(precompb))
            Dt = courant / (maxa / self.hx + maxb / self.hy)
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

        # interpolating averages at cell interfaces
        self.left_face_average_stensil = (
            ConservativeInterpolation.construct_from_order(
                order, "left"
            ).nparray()
        )
        self.right_face_average_stensil = (
            ConservativeInterpolation.construct_from_order(
                order, "right"
            ).nparray()
        )
        self._stensil_size = len(self.left_face_average_stensil)
        self.k = int(np.floor(len(self.left_face_average_stensil) / 2))
        self.gw = self.k + 1

        # interpolating values along cell interfaces
        p = order - 1  # polynomial degree
        q = (
            int(np.ceil((p - 1) / 2)) + 1
        )  # required number of quadrature points
        quadrature_points, weights = np.polynomial.legendre.leggauss(q)
        # scale to cell of width 1
        quadrature_points /= 2
        self.guass_quadrature_weights = weights / 2
        # generater stensils
        list_of_quadrature_stensils = []
        for x in quadrature_points:
            stensil = ConservativeInterpolation.construct_from_order(
                order, x
            ).nparray()
            # if the stensil is short, assume it needs a 0 on either end
            while len(stensil) < self._stensil_size:
                stensil = np.concatenate((np.zeros(1), stensil, np.zeros(1)))
            assert len(stensil) == self._stensil_size
            list_of_quadrature_stensils.append(stensil)
        self.list_of_quadrature_stensils = list_of_quadrature_stensils

        # x and y values at north/south interfaces
        ns_quadrature_points_x = (
            np.tile(self.x, (q, 1))
            + self.hx * quadrature_points[:, np.newaxis]
        )
        ns_quadrature_points_y = np.tile(self.y_interface, (q, 1))
        # x and y values at east/west interfaces
        ew_quadrature_points_x = np.tile(self.x_interface, (q, 1))
        ew_quadrature_points_y = (
            np.tile(self.y, (q, 1))
            + self.hy * quadrature_points[:, np.newaxis]
        )

        # evaluate a at these points
        if callable(a):
            # nsa has shape (2, q, len(y interface), len(x center))
            self.nsa = np.swapaxes(
                np.asarray(
                    [
                        a(x, y)
                        for x, y in zip(
                            ns_quadrature_points_x, ns_quadrature_points_y
                        )
                    ]
                ),
                0,
                1,
            )
            # nsa has shape (2, q, len(y center), len(x interface))
            self.ewa = np.swapaxes(
                np.asarray(
                    [
                        a(x, y)
                        for x, y in zip(
                            ew_quadrature_points_x, ew_quadrature_points_y
                        )
                    ]
                ),
                0,
                1,
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
            if direction == "xy":
                pad_width = gw
            u_with_ghost_cells = np.pad(
                u_without_ghost_cells, pad_width, mode="wrap"
            )
        return u_with_ghost_cells

    def riemann(
        self,
        velocities_at_boundaries: np.ndarray,
        left_of_boundary_values: np.ndarray,
        right_of_boundary_values: np.ndarray,
    ) -> float:
        """
        args:
            velocities_at_boundaries    array of shape (k,m,n)
            left_of_boundary_values     array of shape (k,m,n)
            right_of_boundary_values    array of shape (k,m,n)
        """
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
        gw = self.gw
        negative_gw = -gw
        u_extended = self.apply_bc(u, gw, direction="xy")
        ny, nx = u.shape
        # construct an array of staggered state arrays such that a 3D matrix operation
        # with a stensil multiplies weights by their referenced cells north/south
        # east/west
        ew_stack = stack3d(u_extended, self._stensil_size, direction="x")
        ns_stack = stack3d(u_extended, self._stensil_size, direction="y")
        # 1d face average reconstruction
        ubar_north = apply_stensil(ns_stack, self.right_face_average_stensil)
        ubar_south = apply_stensil(ns_stack, self.left_face_average_stensil)
        ubar_east = apply_stensil(ew_stack, self.right_face_average_stensil)
        ubar_west = apply_stensil(ew_stack, self.left_face_average_stensil)
        # quadrature point reconstruction
        ubar_north_stack = stack3d(
            ubar_north, self._stensil_size, direction="x"
        )
        north_quadrature_interpolations = np.asarray(
            [
                apply_stensil(ubar_north_stack, stensil)
                for stensil in self.list_of_quadrature_stensils
            ]
        )
        ubar_south_stack = stack3d(
            ubar_south, self._stensil_size, direction="x"
        )
        south_quadrature_interpolations = np.asarray(
            [
                apply_stensil(ubar_south_stack, stensil)
                for stensil in self.list_of_quadrature_stensils
            ]
        )
        ubar_east_stack = stack3d(ubar_east, self._stensil_size, direction="y")
        east_quadrature_interpolations = np.asarray(
            [
                apply_stensil(ubar_east_stack, stensil)
                for stensil in self.list_of_quadrature_stensils
            ]
        )
        ubar_west_stack = stack3d(ubar_west, self._stensil_size, direction="y")
        west_quadrature_interpolations = np.asarray(
            [
                apply_stensil(ubar_west_stack, stensil)
                for stensil in self.list_of_quadrature_stensils
            ]
        )
        # reimann problem to solve fluxes at each face
        if callable(self.a):
            ns_quadrature_fluxes = self.riemann(
                self.nsa[1],
                north_quadrature_interpolations[:, :-1, 1:-1],
                south_quadrature_interpolations[:, 1:, 1:-1],
            )
            ew_quadrature_fluxes = self.riemann(
                self.ewa[0],
                east_quadrature_interpolations[:, 1:-1, :-1],
                west_quadrature_interpolations[:, 1:-1, 1:],
            )
            # evaluate integral
            ns_fluxes = apply_stensil(
                ns_quadrature_fluxes, self.guass_quadrature_weights
            )
            ew_fluxes = apply_stensil(
                ew_quadrature_fluxes, self.guass_quadrature_weights
            )
        else:
            ew_fluxes = self.riemann(
                self.a,
                ubar_east[gw:negative_gw, :-1],
                ubar_west[gw:negative_gw, 1:],
            )
            ns_fluxes = self.riemann(
                self.b,
                ubar_north[:-1, gw:negative_gw],
                ubar_south[1:, gw:negative_gw],
            )

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
            return np.sum(np.abs(approx - truth) * self.hx * self.hy)
        if norm == "l2":
            return np.sqrt(
                np.sum(np.power(approx - truth, 2)) * self.hx * self.hy
            )
        if norm == "inf":
            return np.max(np.abs(approx - truth))

    def plot(self):
        # global max and min
        hmin, hmax = np.min(self.u), np.max(self.u)
        bounds = [self.x[0], self.x[-1], self.y[0], self.y[-1]]

        # select plot frames
        data1 = np.flipud(self.u[0])
        data2 = np.flipud(self.u[int(0.25 * self.loglen)])
        data3 = np.flipud(self.u[int(0.5 * self.loglen)])
        data4 = np.flipud(self.u[int(0.75 * self.loglen)])
        data5 = np.flipud(self.u[-1])
        data6 = np.flipud(self.u[-1] - self.u[0])

        # 2x3 grid of subplots
        fig = plt.figure(figsize=(10, 6.5))
        gs = gridspec.GridSpec(
            nrows=2,
            ncols=4,
            width_ratios=[1, 1, 1, 0.1],
            height_ratios=[1, 1],
            figure=fig,
        )
        axs = gs.subplots()
        im1 = axs[0, 0].imshow(
            data1, cmap="cool", vmin=hmin, vmax=hmax, extent=bounds
        )
        axs[0, 1].imshow(
            data2, cmap="cool", vmin=hmin, vmax=hmax, extent=bounds
        )
        axs[0, 2].imshow(
            data3, cmap="cool", vmin=hmin, vmax=hmax, extent=bounds
        )
        axs[1, 0].imshow(
            data4, cmap="cool", vmin=hmin, vmax=hmax, extent=bounds
        )
        axs[1, 1].imshow(
            data5, cmap="cool", vmin=hmin, vmax=hmax, extent=bounds
        )
        im6 = axs[1, 2].imshow(data6, cmap="hot", extent=bounds)

        # color bars
        cbar1 = fig.add_subplot(gs[0, 3])
        plt.colorbar(im1, cax=cbar1)
        cbar1.yaxis.set_ticks_position("right")
        cbar2 = fig.add_subplot(gs[1, 3])
        plt.colorbar(im6, cax=cbar2)
        cbar2.yaxis.set_ticks_position("right")

        # Add titles to the plots
        axs[0, 0].set_title(f"t = {self.t[0]:.3f}")
        axs[0, 1].set_title(f"t = {self.t[int(0.25 * self.loglen)]:.3f}")
        axs[0, 2].set_title(f"t = {self.t[int(0.5 * self.loglen)]:.3f}")
        axs[1, 0].set_title(f"t = {self.t[int(0.75 * self.loglen)]:.3f}")
        axs[1, 1].set_title(f"t = {self.t[-1]:.3f}")
        axs[1, 2].set_title(
            f"u(t = {self.t[-1]:.3f}) - u(t = {self.t[0]:.3f})"
        )

        # Add a title to the entire figure
        fig.suptitle("Heatmap Grid")

        # Show the plot
        plt.show()
