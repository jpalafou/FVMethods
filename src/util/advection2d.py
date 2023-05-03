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
        v: tuple = (1, 1),
        courant: float = 0.5,
        order: int = 1,
        bc_type: str = "periodic",
        apriori_limiting: str = None,
        loglen: int = 21,
        adujst_time_step: bool = False,
    ):
        # spatial discretization in x
        self.n = (n, n) if isinstance(n, int) else n
        ny, nx = self.n
        self.x_interface = np.linspace(
            x[0], x[1], num=nx + 1
        )  # cell interfaces
        self.x = 0.5 * (
            self.x_interface[:-1] + self.x_interface[1:]
        )  # cell centers
        self.Lx = x[1] - x[0]  # domain size in x
        self.hx = self.Lx / nx  # cell size in x
        self.hx_recip = nx / self.Lx  # 1 / h

        # spatial discretization in y
        y = x if y is None else y
        self.y_interface = np.linspace(
            y[0], y[1], num=ny + 1
        )  # cell interfaces
        self.y = 0.5 * (
            self.y_interface[:-1] + self.y_interface[1:]
        )  # cell centers
        self.Ly = y[1] - y[0]  # domain size in y
        self.hy = self.Ly / ny  # cell size in y
        self.hy_recip = ny / self.Ly  # 1 / h

        # maximum expected advection velocities
        if isinstance(v, tuple):
            vx_max, vy_max = abs(v[0]), abs(v[1])
        if callable(v):
            self.v = v
            # precompute velocity at cell corners and use this as estimate for max
            xx_interface, yy_interface = np.meshgrid(
                self.x_interface, self.y_interface
            )
            v_max = v(xx_interface, yy_interface)
            vx_max, vy_max = np.max(abs(v_max[0])), np.max(abs(v_max[1]))

        # time discretization
        self.courant = courant
        self.adujst_time_step = adujst_time_step
        self.order = order
        Dt = courant / (vx_max / self.hx + vy_max / self.hy)
        Dt_adjustment = None
        if adujst_time_step and order > 4:
            Dt_adjustment_x = rk4_Dt_adjust(self.hx, self.Lx, order)
            Dt_adjustment_y = rk4_Dt_adjust(self.hy, self.Ly, order)
            Dt_adjustment = min(Dt_adjustment_x, Dt_adjustment_y)
            Dt = Dt * Dt_adjustment
            print(
                f"Decreasing timestep by a factor of {Dt_adjustment} to maintain",
                f" order {order} with rk4",
            )
        self.Dt_adjustment = Dt_adjustment
        # round to nearest integer number of timesteps
        n_timesteps = int(np.ceil(T / Dt))
        self.Dt = T / n_timesteps
        self.t = np.linspace(0, T, num=n_timesteps + 1)

        # initial/boundary conditions
        u0 = initial_condition2d(self.x, self.y, u0_preset)
        self.bc_type = bc_type

        # initialize integrator
        super().__init__(u0=u0, t=self.t, loglen=loglen)

        # right/left interpolation from a volume or line segment
        left_interface_stensil = (
            ConservativeInterpolation.construct_from_order(
                order, "left"
            ).nparray()
        )
        right_interface_stensil = (
            ConservativeInterpolation.construct_from_order(
                order, "right"
            ).nparray()
        )
        self._stensil_size = len(left_interface_stensil)
        self.k = int(
            np.floor(len(left_interface_stensil) / 2)
        )  # cell reach of stensil
        self.gw = self.k + 1  # ghost width

        # quadrature points setup
        p = order - 1  # degree of reconstructed polynomial
        N_G = int(
            np.ceil((p + 1) / 2)
        )  # number of gauss-legendre quadrature points
        N_GL = int(
            np.ceil((p + 3) / 2)
        )  # number of gauss-lobatto quadrature points

        # stensils for reconstructing the average along a line segment within a cell
        self.list_of_line_stensils = []
        # guass-legendre quadrature
        (
            gauss_quadr_points,
            gauss_quadr_weights,
        ) = np.polynomial.legendre.leggauss(N_G)
        # transform to cell coordinate
        gauss_quadr_points /= 2
        gauss_quadr_weights /= 2
        self.gauss_quadr_weights = (
            gauss_quadr_weights  # for evaluating line integrals
        )
        # reconstruct polynomial and evaluate at each quadrature point
        for x in gauss_quadr_points:
            stensil = ConservativeInterpolation.construct_from_order(
                order, x
            ).nparray()
            # if the stensil is short, assume it needs a 0 on either end
            while len(stensil) < self._stensil_size:
                stensil = np.concatenate((np.zeros(1), stensil, np.zeros(1)))
            assert len(stensil) == self._stensil_size
            self.list_of_line_stensils.append(
                stensil
            )  # ordered from left to right

        # stensils for reconstructing pointwise values along a line average
        self.apriori_limiting = apriori_limiting if order > 1 else None
        list_of_GL_stensils = []
        if self.apriori_limiting is not None and N_GL > 2:
            # interpolating values along line segments
            (
                interior_GL_quadr_points,
                _,
            ) = np.polynomial.legendre.leggauss(N_GL - 2)
            endpoint_GL_quadr_weights = 2 / (N_GL * (N_GL - 1))
            # scale to cell of width 1
            interior_GL_quadr_points /= 2
            endpoint_GL_quadr_weights /= 2
            # mpp lite
            if self.apriori_limiting == "mpp lite":
                interior_GL_quadr_points = np.array([])
            # generate stensils two are given
            for x in interior_GL_quadr_points:
                stensil = ConservativeInterpolation.construct_from_order(
                    order, x
                ).nparray()
                # if the stensil is short, assume it needs a 0 on either end
                while len(stensil) < self._stensil_size:
                    stensil = np.concatenate(
                        (np.zeros(1), stensil, np.zeros(1))
                    )
                assert len(stensil) == self._stensil_size
                list_of_GL_stensils.append(stensil)
            # check if timestep is small enough for mpp
            if (
                self.Dt * (vx_max / self.hx + vy_max / self.hy)
                > endpoint_GL_quadr_weights
            ):
                print(
                    "WARNING: Maximum principle preserving not satisfied.\nTry a ",
                    f"courant condition less than {endpoint_GL_quadr_weights}\n",
                )

        # assort list of stensils for pointwise interpolation
        self.list_of_pointwise_stensils = (
            [left_interface_stensil]
            + list_of_GL_stensils
            + [right_interface_stensil]
        )

        # x and y values at quadrature points
        na = np.newaxis
        # quadrature points along east/west interfaces, shape (N_G, ny, nx + 1)
        EW_qaudr_points_x = np.tile(self.x_interface, (N_G, ny, 1))
        EW_qaudr_points_y = (
            np.tile(np.tile(self.y, (nx + 1, 1)).T, (N_G, 1, 1))
            + self.hy * gauss_quadr_points[:, na, na]
        )
        # quadrature points along north/south interfaces, shape (N_G, ny + 1, nx)
        NS_qaudr_points_x = (
            np.tile(self.x, (N_G, ny + 1, 1))
            + self.hx * gauss_quadr_points[:, na, na]
        )
        NS_qaudr_points_y = np.tile(
            np.tile(self.y_interface, (nx, 1)).T, (N_G, 1, 1)
        )

        # evaluate v at the interfaces
        # only store component of velocity that is normal to interface
        if isinstance(v, tuple):
            self.v_EW_interface = v[0] * np.ones(EW_qaudr_points_x.shape)
            self.v_NS_interface = v[1] * np.ones(NS_qaudr_points_x.shape)
        if callable(v):
            self.v_EW_interface = v(EW_qaudr_points_x, EW_qaudr_points_y)[0]
            self.v_NS_interface = v(NS_qaudr_points_x, NS_qaudr_points_y)[1]

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
        v: np.ndarray,
        left_value: np.ndarray,
        right_value: np.ndarray,
    ) -> float:
        """
        args:
            arrays of shape (k,m,n)
            v   advection velocity defined at an interface
            left_value  value to the left of the interface
            right_value value to the right of the interface
        returns:
            array of shape (k,m,n)
            fluxes at interface chosen based on advection direction
        """
        left_flux, right_flux = v * left_value, v * right_value
        return (
            (right_flux + left_flux) - np.abs(v) * (right_value - left_value)
        ) / 2.0

    def udot(self, u: np.ndarray, t_i: float, dt: float = None) -> np.ndarray:
        # construct an extended array with ghost zones and apply bc
        u_extended = self.apply_bc(u, self.gw, direction="xy")
        # stack volume arrays NS/EW to prepare for guass stensil operation
        EW_stack = stack3d(u_extended, self._stensil_size, direction="x")
        NS_stack = stack3d(u_extended, self._stensil_size, direction="y")
        # line average reconstruction, first dimension is guass-legendre point
        vertical_line_averages = np.asarray(
            [
                apply_stensil(EW_stack, stensil)
                for stensil in self.list_of_line_stensils
            ]
        )
        horizontal_line_averages = np.asarray(
            [
                apply_stensil(NS_stack, stensil)
                for stensil in self.list_of_line_stensils
            ]
        )
        # stack line average arrays to prepare for guass-lobatto stensil operation
        EW_stacks = np.asarray(
            [
                stack3d(i, self._stensil_size, direction="x")
                for i in horizontal_line_averages
            ]
        )
        NS_stacks = np.asarray(
            [
                stack3d(i, self._stensil_size, direction="y")
                for i in vertical_line_averages
            ]
        )
        # pointwise reconstruction, first dimension is guass-legendre point
        # second dimension is guass-lobatto point
        horizontal_points = np.asarray(
            [
                [
                    apply_stensil(EW_stack, stensil)
                    for stensil in self.list_of_pointwise_stensils
                ]
                for EW_stack in EW_stacks
            ]
        )
        vertical_points = np.asarray(
            [
                [
                    apply_stensil(NS_stack, stensil)
                    for stensil in self.list_of_pointwise_stensils
                ]
                for NS_stack in NS_stacks
            ]
        )
        north_points = vertical_points[:, -1, :, :]
        south_points = vertical_points[:, 0, :, :]
        east_points = horizontal_points[:, -1, :, :]
        west_points = horizontal_points[:, 0, :, :]
        # find slope limiter
        if self.apriori_limiting is not None:
            # max and min of 4 neighbors
            u_2gw = self.apply_bc(u, 2, direction="xy")
            u_1gw = u_2gw[1:-1, 1:-1]
            list_of_9_neighbors = [
                u_1gw,
                u_2gw[:-2, 1:-1],
                u_2gw[2:, 1:-1],
                u_2gw[1:-1, :-2],
                u_2gw[1:-1, 2:],
                u_2gw[2:, 2:],
                u_2gw[2:, :-2],
                u_2gw[:-2, 2:],
                u_2gw[:-2, :-2],
            ]
            M = np.maximum.reduce(list_of_9_neighbors)
            m = np.minimum.reduce(
                list_of_9_neighbors + [1e-12 * np.ones(u_1gw.shape)]
            )
            # max and min of u evaluated at quadrature points
            M_ij = np.maximum(
                np.amax(horizontal_points, axis=(0, 1)),
                np.amax(vertical_points, axis=(0, 1)),
            )
            m_ij = np.minimum(
                np.amin(horizontal_points, axis=(0, 1)),
                np.amin(vertical_points, axis=(0, 1)),
            )
            # evaluate slope limiter
            theta = np.ones(u_1gw.shape)
            M_arg = np.abs(M - u_1gw) / (np.abs(M_ij - u_1gw) + 1e-6)
            m_arg = np.abs(m - u_1gw) / (np.abs(m_ij - u_1gw) + 1e-6)
            theta = np.where(M_arg < theta, M_arg, theta)
            theta = np.where(m_arg < theta, m_arg, theta)
            # limit flux points
            _k = self.k
            _minus_k = -self.k
            horizontal_fallback = horizontal_line_averages[:, :, _k:_minus_k]
            vertical_fallback = vertical_line_averages[:, _k:_minus_k, :]
            north_points = (
                theta[np.newaxis, ...] * (north_points - vertical_fallback)
                + vertical_fallback
            )
            south_points = (
                theta[np.newaxis, ...] * (south_points - vertical_fallback)
                + vertical_fallback
            )
            east_points = (
                theta[np.newaxis, ...] * (east_points - horizontal_fallback)
                + horizontal_fallback
            )
            west_points = (
                theta[np.newaxis, ...] * (west_points - horizontal_fallback)
                + horizontal_fallback
            )
        # reimann problem to solve fluxes at each face
        NS_pointwise_fluxes = self.riemann(
            self.v_NS_interface,
            north_points[:, :-1, 1:-1],
            south_points[:, 1:, 1:-1],
        )
        EW_pointwise_fluxes = self.riemann(
            self.v_EW_interface,
            east_points[:, 1:-1, :-1],
            west_points[:, 1:-1, 1:],
        )
        # guass quadrature integral approximation
        NS_fluxes = apply_stensil(
            NS_pointwise_fluxes, self.gauss_quadr_weights
        )
        EW_fluxes = apply_stensil(
            EW_pointwise_fluxes, self.gauss_quadr_weights
        )
        # spatial derivative operator
        dudt = -self.hx_recip * (
            EW_fluxes[:, 1:] - EW_fluxes[:, :-1]
        ) + -self.hy_recip * (NS_fluxes[1:, :] - NS_fluxes[:-1, :])
        return dudt

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
