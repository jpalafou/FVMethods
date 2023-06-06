# advection solution schems
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from finite_volume.initial_conditions import generate_ic
from finite_volume.integrate import Integrator
from finite_volume.fvscheme import ConservativeInterpolation, TransverseIntegral
from finite_volume.aposteriori.simple_trouble_detection2d import (
    trouble_detection2d,
    compute_second_order_fluxes,
)


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
    stensilcopy = stensil.copy() if stensil.ndim == 3 else stensil.reshape(-1, 1, 1)
    return np.sum(u * stensilcopy, axis=0) / sum(stensilcopy)


def trim2d(u: np.ndarray, axis: tuple, cut_length: tuple):
    """
    args:
        u           np array of arbitrary dimension
        axis        tuple or int
        cut_length  same length as axis
    returns:
        u           symmetrically cut at the ends along axis by cut_length
    """
    if isinstance(axis, int):
        axis = (axis,)  # Convert single axis to a tuple
    if isinstance(cut_length, int):
        cut_length = (cut_length,) * len(axis)  # Convert cut_length to a tuple
    slices = [slice(None)] * u.ndim
    for ax, cl in zip(axis, cut_length):
        if cl != 0:
            slices[ax] = slice(cl, -cl)
    return u[tuple(slices)]


# class definition
class AdvectionSolver(Integrator):
    """
    args:
        u0  preset string or callable function describing solution at t=0
        n   tuple of number of cells in x and y
        x   tuple of boundaries in x
        y   tuple of boundaries in y
        T   solving time
        v   tuple of floating point velocity components
            callable function of x and y
        courant                 stability condition
        order                   accuracy requirement for polynomial interpolation
        bc                      string describing a pre-coded boudnary condition
        flux_strategy           'gauss-legendre' or 'transverse'
        apriori_limiting        None, 'mpp', or 'mpp lite'
        aposteriori_limiting    bool
        loglen                  number of saved states
        adujst_time_step        whether to reduce timestep for order >4
    returns:
        u   array of saved states
    """

    def __init__(
        self,
        u0: str = "square",
        n: tuple = (32, 32),
        x: tuple = (0, 1),
        y: tuple = None,
        T: float = 1,
        v: tuple = (1, 1),
        courant: float = 0.5,
        order: int = 1,
        bc: str = "periodic",
        flux_strategy: str = "gauss-legendre",
        apriori_limiting: str = None,
        aposteriori_limiting: bool = False,
        loglen: int = 21,
        adujst_time_step: bool = False,
    ):
        # spatial discretization in x
        self.n = (n, n) if isinstance(n, int) else n
        ny, nx = self.n
        self.x_interface = np.linspace(x[0], x[1], num=nx + 1)  # cell interfaces
        self.x = 0.5 * (self.x_interface[:-1] + self.x_interface[1:])  # cell centers
        self.Lx = x[1] - x[0]  # domain size in x
        self.hx = self.Lx / nx  # cell size in x
        self.hx_recip = nx / self.Lx  # 1 / h

        # spatial discretization in y
        y = x if y is None else y
        self.y_interface = np.linspace(y[0], y[1], num=ny + 1)  # cell interfaces
        self.y = 0.5 * (self.y_interface[:-1] + self.y_interface[1:])  # cell centers
        self.Ly = y[1] - y[0]  # domain size in y
        self.hy = self.Ly / ny  # cell size in y
        self.hy_recip = ny / self.Ly  # 1 / h

        # maximum expected advection velocities
        if isinstance(v, tuple):
            vx_max, vy_max = abs(v[0]), abs(v[1])
        elif callable(v):
            self.v = v
            # precompute velocity at cell corners and use this as estimate for max
            xx_interface, yy_interface = np.meshgrid(self.x_interface, self.y_interface)
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
        if isinstance(u0, str):
            u0 = generate_ic(type=u0, x=self.x, y=self.y)
        if callable(u0):
            u0 = u0(x=self.x, y=self.y)
        self.bc = bc

        # initialize integrator
        super().__init__(u0=u0, t=self.t, loglen=loglen)

        # stensils: right/left interpolation from a volume or line segment
        left_interface_stensil = ConservativeInterpolation.construct_from_order(
            order, "left"
        ).nparray()
        right_interface_stensil = ConservativeInterpolation.construct_from_order(
            order, "right"
        ).nparray()
        self._conservative_stensil_size = len(left_interface_stensil)
        self._conservative_k = int(
            np.floor(len(left_interface_stensil) / 2)
        )  # cell reach of stensil

        # quadrature points setup
        p = order - 1  # degree of reconstructed polynomial
        N_G = int(np.ceil((p + 1) / 2))  # number of gauss-legendre quadrature points
        N_G = N_G + 1 if N_G % 2 == 0 and flux_strategy == "transverse" else N_G
        N_GL = int(np.ceil((p + 3) / 2))  # number of gauss-lobatto quadrature points

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
        self.gauss_quadr_weights = gauss_quadr_weights  # for evaluating line integrals
        # reconstruct polynomial and evaluate at each quadrature point
        for x in gauss_quadr_points:
            stensil = ConservativeInterpolation.construct_from_order(order, x).nparray()
            # if the stensil is short, assume it needs a 0 on either end
            while len(stensil) < self._conservative_stensil_size:
                stensil = np.concatenate((np.zeros(1), stensil, np.zeros(1)))
            assert len(stensil) == self._conservative_stensil_size
            self.list_of_line_stensils.append(stensil)  # ordered from left to right

        # simplified list of line stensils for transverse flux
        if flux_strategy == "transverse":
            self.left_interface_stensil = left_interface_stensil
            self.right_interface_stensil = right_interface_stensil
            self.cell_center_stensil = self.list_of_line_stensils[N_G // 2]

        # stensils for reconstructing pointwise values along a line average
        self.apriori_limiting = apriori_limiting
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
                while len(stensil) < self._conservative_stensil_size:
                    stensil = np.concatenate((np.zeros(1), stensil, np.zeros(1)))
                assert len(stensil) == self._conservative_stensil_size
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
        if flux_strategy == "gauss-legendre":
            self.list_of_pointwise_stensils = (
                [left_interface_stensil]
                + list_of_GL_stensils
                + [right_interface_stensil]
            )
        if flux_strategy == "transverse":
            self.list_of_pointwise_stensils = [left_interface_stensil] + [
                right_interface_stensil
            ]

        # transverse integral stensil
        if flux_strategy == "transverse":
            self.integral_stensil = TransverseIntegral.construct_from_order(
                order
            ).nparray()
            self._transverse_stensil_size = len(self.integral_stensil)
            self._transverse_k = int(
                np.floor(len(self.integral_stensil) / 2)
            )  # cell reach of stensil

        # ghost width
        if flux_strategy == "gauss-legendre":
            self.gw = self._conservative_k + 1
        elif flux_strategy == "transverse":
            self.gw = self._transverse_k + self._conservative_k + 1

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
            np.tile(self.x, (N_G, ny + 1, 1)) + self.hx * gauss_quadr_points[:, na, na]
        )
        NS_qaudr_points_y = np.tile(np.tile(self.y_interface, (nx, 1)).T, (N_G, 1, 1))

        # evaluate v at the interfaces
        # only store component of velocity that is normal to interface
        if isinstance(v, tuple):
            self.v_EW_interface = v[0] * np.ones(EW_qaudr_points_x.shape)
            self.v_NS_interface = v[1] * np.ones(NS_qaudr_points_x.shape)
        if callable(v):
            self.v_EW_interface = v(EW_qaudr_points_x, EW_qaudr_points_y)[0]
            self.v_NS_interface = v(NS_qaudr_points_x, NS_qaudr_points_y)[1]

        # initialize a posteriori limiting
        self.aposteriori_limiting = aposteriori_limiting
        # define normal velocity at midpoints of cell interfaces
        if isinstance(v, tuple):
            self.v_EW_midpoints = v[0] * np.ones((ny, nx + 1))
            self.v_NS_midpoints = v[1] * np.ones((ny + 1, nx))
        if callable(v):
            x_EW_midpoints, y_EW_midpoints = np.meshgrid(self.x_interface, self.y)
            x_NS_midpoints, y_NS_midpoints = np.meshgrid(self.x, self.y_interface)
            self.v_EW_midpoints = v(x_EW_midpoints, y_EW_midpoints)[0]
            self.v_NS_midpoints = v(x_NS_midpoints, y_NS_midpoints)[1]
        if flux_strategy == "transverse":
            self.v_EW_midpoints_gw = self.apply_bc(
                self.v_EW_midpoints, self._transverse_k, "y"
            )
            self.v_NS_midpoints_gw = self.apply_bc(
                self.v_NS_midpoints, self._transverse_k, "x"
            )

        # fluxes
        self.NS_fluxes = np.zeros((ny + 1, nx))
        self.EW_fluxes = np.zeros((ny, nx + 1))

        # dynamic function assignment
        self.flux_strategy = flux_strategy
        if self.flux_strategy == "gauss-legendre":
            self.compute_high_order_fluxes = self.legendre_fluxes
        elif self.flux_strategy == "transverse":
            self.compute_high_order_fluxes = self.transverse_fluxes

        if self.apriori_limiting is None:
            self.apriori_limiter = self.no_limiter
        else:
            self.apriori_limiter = self.mpp_limiter

        if self.aposteriori_limiting:
            self.aposteriori_limiter = self.revise_solution
        else:
            self.aposteriori_limiter = self.dont_revise_solution

    def udot(self, u: np.ndarray, t_i: float = None, dt: float = None) -> np.ndarray:
        """
        args:
            u       (ny, nx)
            t_i     time at which u is defined
            dt      timestep size from t_i to t_i+1
        returns:
            dudt    (ny, nx) at t_i
        """
        self.compute_high_order_fluxes(u=u)
        self.aposteriori_limiter(u0=u, dt=dt)
        return self.get_dynamics()

    def apply_bc(
        self, u_without_ghost_cells: np.ndarray, gw: int, dim: str
    ) -> np.ndarray:
        """
        args:
            u_without_ghost_cells   (ny, nx)
            gw  number of ghost cells on either side of u
            dim   'x', 'y', or 'xy'
        returns:
            u_with_ghost_cells  (ny, nx + 2 * gw) if dim = 'x'
                                (ny + 2 * gw, nx) if dim = 'y'
                                (ny + 2 * gw, nx + 2 * gw) if dim = 'xy'
        """
        if self.bc == "periodic":
            if dim == "x":
                pad_width = ((0, 0), (gw, gw))
            if dim == "y":
                pad_width = ((gw, gw), (0, 0))
            if dim == "xy":
                pad_width = gw
            u_with_ghost_cells = np.pad(u_without_ghost_cells, pad_width, mode="wrap")
        return u_with_ghost_cells

    def legendre_fluxes(self, u: np.ndarray):
        """
        compute fluxes with the Gauss-Legendre quadrature method
        args:
            u   array of volume averages (ny, nx)
        overwrites:
            self.NS_fluxes  (ny + 1, nx)
            self.EW_fluxes  (ny, nx + 1)
        """
        # construct an extended array with ghost zones and apply bc
        u_extended = self.apply_bc(u, gw=self.gw, dim="xy")
        # stack volume arrays NS/EW to prepare for guass stensil operation
        EW_stack = stack3d(u_extended, self._conservative_stensil_size, direction="x")
        NS_stack = stack3d(u_extended, self._conservative_stensil_size, direction="y")
        # line average reconstruction, first dimension is guass-legendre point
        vertical_line_averages = np.asarray(
            [apply_stensil(EW_stack, stensil) for stensil in self.list_of_line_stensils]
        )
        horizontal_line_averages = np.asarray(
            [apply_stensil(NS_stack, stensil) for stensil in self.list_of_line_stensils]
        )
        # stack line average arrays to prepare for guass-lobatto stensil operation
        EW_stacks = np.asarray(
            [
                stack3d(i, self._conservative_stensil_size, direction="x")
                for i in horizontal_line_averages
            ]
        )
        NS_stacks = np.asarray(
            [
                stack3d(i, self._conservative_stensil_size, direction="y")
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
        # apply slope limiter
        (north_points, south_points, east_points, west_points,) = self.apriori_limiter(
            horizontal_points,
            vertical_points,
            trim2d(horizontal_line_averages, axis=2, cut_length=self._conservative_k),
            trim2d(vertical_line_averages, axis=1, cut_length=self._conservative_k),
            u,
            gw=1,
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
        self.NS_fluxes[...] = apply_stensil(
            NS_pointwise_fluxes, self.gauss_quadr_weights
        )
        self.EW_fluxes[...] = apply_stensil(
            EW_pointwise_fluxes, self.gauss_quadr_weights
        )

    def transverse_fluxes(self, u: np.ndarray):
        """
        compute fluxes with the transverse method
        args:
            u   array of volume averages (ny, nx)
        overwrites:
            self.NS_fluxes  (ny + 1, nx)
            self.EW_fluxes  (ny, nx + 1)s
        """
        # construct an extended array with ghost zones and apply bc
        u_extended = self.apply_bc(u, gw=self.gw, dim="xy")
        # stack volume arrays NS/EW to prepare for guass stensil operation
        EW_stack = stack3d(u_extended, self._conservative_stensil_size, direction="x")
        NS_stack = stack3d(u_extended, self._conservative_stensil_size, direction="y")
        # line average reconstruction
        vertical_line_average = apply_stensil(EW_stack, self.cell_center_stensil)
        horizontal_line_average = apply_stensil(NS_stack, self.cell_center_stensil)
        # stack for endpoint reconstruction
        EW_stack = stack3d(
            horizontal_line_average, self._conservative_stensil_size, direction="x"
        )
        NS_stack = stack3d(
            vertical_line_average, self._conservative_stensil_size, direction="y"
        )
        # pointwise reconstruction
        horizontal_points = np.asarray(
            [
                apply_stensil(EW_stack, stensil)
                for stensil in self.list_of_pointwise_stensils
            ]
        )
        vertical_points = np.asarray(
            [
                apply_stensil(NS_stack, stensil)
                for stensil in self.list_of_pointwise_stensils
            ]
        )
        # apply slope limiter
        (north_points, south_points, east_points, west_points,) = self.apriori_limiter(
            horizontal_points[np.newaxis],
            vertical_points[np.newaxis],
            trim2d(horizontal_line_average, axis=1, cut_length=self._conservative_k)[
                np.newaxis
            ],
            trim2d(vertical_line_average, axis=0, cut_length=self._conservative_k)[
                np.newaxis
            ],
            u,
            gw=self._transverse_k + 1,
        )
        # reimann problem to solve fluxes at each face
        NS_pointwise_fluxes = self.riemann(
            self.v_NS_midpoints_gw,
            trim2d(north_points[0], axis=0, cut_length=self._transverse_k)[:-1, 1:-1],
            trim2d(south_points[0], axis=0, cut_length=self._transverse_k)[1:, 1:-1],
        )
        EW_pointwise_fluxes = self.riemann(
            self.v_EW_midpoints_gw,
            trim2d(east_points[0], axis=1, cut_length=self._transverse_k)[1:-1, :-1],
            trim2d(west_points[0], axis=1, cut_length=self._transverse_k)[1:-1, 1:],
        )
        # transverse integral stensil
        EW_stack = stack3d(NS_pointwise_fluxes, self._transverse_stensil_size, "x")
        NS_stack = stack3d(EW_pointwise_fluxes, self._transverse_stensil_size, "y")
        self.NS_fluxes[...] = apply_stensil(EW_stack, self.integral_stensil)
        self.EW_fluxes[...] = apply_stensil(NS_stack, self.integral_stensil)

    def no_limiter(
        self,
        horizontal_points: np.ndarray,
        vertical_points: np.ndarray,
        horizontal_line_averages: np.ndarray,
        vertical_line_averages: np.ndarray,
        u: np.ndarray,
        gw: int,
    ):
        """
        return mpp slope limited pointwise interpolations along cell borders
        args:
            horizontal_points           (# horizontal lines, # points per line,
                                        ny + 2, nx + 2)
            vertical_points             (# vertical lines, # points per line,
                                        ny + 2, nx + 2)
            horizontal_line_averages    (# horizontal lines, ny + 2, nx + 2 * gw)
            vertical_line_averages      (# vertical lines, ny + 2 * gw, nx + 2)
            u                           (ny, nx)
            gw                          # of ghost cells entering the slope limiter
        returns:
            north_points    (# vertical lines, ny + 2, nx + 2)
            south_points    (# vertical lines, ny + 2, nx + 2)
            east_points     (# horizontal lines, ny + 2, nx + 2)
            west_points     (# horizontal lines, ny + 2, nx + 2)
        """
        north_points = vertical_points[:, -1, :, :]
        south_points = vertical_points[:, 0, :, :]
        east_points = horizontal_points[:, -1, :, :]
        west_points = horizontal_points[:, 0, :, :]
        return north_points, south_points, east_points, west_points

    def mpp_limiter(
        self,
        horizontal_points: np.ndarray,
        vertical_points: np.ndarray,
        horizontal_line_averages: np.ndarray,
        vertical_line_averages: np.ndarray,
        u: np.ndarray,
        gw: int,
    ):
        """
        return mpp slope limited pointwise interpolations along cell borders
        args:
            horizontal_points           (# horizontal lines, # points per line,
                                        ny + 2, nx + 2)
            vertical_points             (# vertical lines, # points per line,
                                        ny + 2, nx + 2)
            horizontal_line_averages    (# horizontal lines, ny + 2, nx + 2 * gw)
            vertical_line_averages      (# vertical lines, ny + 2 * gw, nx + 2)
            u                           (ny, nx)
            gw                          # of ghost cells entering the slope limiter
        returns:
            north_points    (# vertical lines, ny + 2, nx + 2)
            south_points    (# vertical lines, ny + 2, nx + 2)
            east_points     (# horizontal lines, ny + 2, nx + 2)
            west_points     (# horizontal lines, ny + 2, nx + 2)
        """
        north_points, south_points, east_points, west_points = self.no_limiter(
            horizontal_points,
            vertical_points,
            horizontal_line_averages,
            vertical_line_averages,
            u,
            gw,
        )
        # max and min of 9 neighbors
        u_gw_plus_1 = self.apply_bc(u, gw=gw + 1, dim="xy")  # u with a ghost width of 2
        u_gw = u_gw_plus_1[1:-1, 1:-1]  # u with a ghost width of 1
        list_of_9_neighbors = [
            u_gw,
            u_gw_plus_1[:-2, 1:-1],
            u_gw_plus_1[2:, 1:-1],
            u_gw_plus_1[1:-1, :-2],
            u_gw_plus_1[1:-1, 2:],
            u_gw_plus_1[2:, 2:],
            u_gw_plus_1[2:, :-2],
            u_gw_plus_1[:-2, 2:],
            u_gw_plus_1[:-2, :-2],
        ]
        M = np.maximum.reduce(list_of_9_neighbors)
        m = np.minimum.reduce(list_of_9_neighbors + [1e-12 * np.ones(u_gw.shape)])
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
        theta = np.ones(u_gw.shape)
        M_arg = np.abs(M - u_gw) / ((np.abs(M_ij - u_gw) + 1e-12))
        m_arg = np.abs(m - u_gw) / ((np.abs(m_ij - u_gw) + 1e-12))
        theta = np.where(M_arg < theta, M_arg, theta)
        theta = np.where(m_arg < theta, m_arg, theta)
        # limit flux points
        horizontal_fallback = horizontal_line_averages
        vertical_fallback = vertical_line_averages
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
        return north_points, south_points, east_points, west_points

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
            pointwise fluxes at interface chosen based on advection direction
        """
        left_flux, right_flux = v * left_value, v * right_value
        return ((right_flux + left_flux) - np.abs(v) * (right_value - left_value)) / 2.0

    def revise_solution(self, u0: np.ndarray, dt: float):
        """
        args:
            u0  initial state (ny, nx)
            ucandidate  proposed next state (ny, nx)
        overwrites:
            self.NS_fluxes and self.EW_fluxes to fall back to second order
            when trouble is detected
        """
        # compute candidate solution
        ucandidate = u0 + dt * self.get_dynamics()
        # give the previous and candidate solutions two ghost cells
        u0_2gw = self.apply_bc(u0, 2, dim="xy")
        ucandidate_2gw = self.apply_bc(ucandidate, 2, dim="xy")
        # reshape into 3d array to match david's code
        u0_2gw = u0_2gw[np.newaxis]
        ucandidate_2gw = ucandidate_2gw[np.newaxis]
        # find troubled cells
        troubled_cells = trouble_detection2d(
            u0_2gw, ucandidate_2gw, hx=self.hx, hy=self.hy
        )
        NS_troubled_faces = np.zeros(self.NS_fluxes.shape)[np.newaxis]
        EW_troubled_faces = np.zeros(self.EW_fluxes.shape)[np.newaxis]

        # find troubled faces
        if np.any(troubled_cells):
            EW_troubled_faces[:, :, :-1] = troubled_cells
            EW_troubled_faces[:, :, 1:] = np.where(
                troubled_cells == 1, 1, EW_troubled_faces[:, :, 1:]
            )
            NS_troubled_faces[:, :-1, :] = troubled_cells
            NS_troubled_faces[:, 1:, :] = np.where(
                troubled_cells == 1, 1, NS_troubled_faces[:, 1:, :]
            )
            # find 2nd order pointwise interpolations at cell centers
            west_midpoint, east_midpoint = compute_second_order_fluxes(u0_2gw, dim="x")
            south_midpoint, north_midpoint = compute_second_order_fluxes(
                u0_2gw, dim="y"
            )

            # compute fluxes using face midpoints
            NS_fluxes_2nd_order = self.riemann(
                self.v_NS_midpoints[np.newaxis],
                north_midpoint[:, :-1, :],
                south_midpoint[:, 1:, :],
            )
            EW_fluxes_2nd_order = self.riemann(
                self.v_EW_midpoints[np.newaxis],
                east_midpoint[:, :, :-1],
                west_midpoint[:, :, 1:],
            )

            # revise fluxes
            self.NS_fluxes = np.where(
                NS_troubled_faces[0] == 1,
                NS_fluxes_2nd_order[0],
                self.NS_fluxes,
            )
            self.EW_fluxes = np.where(
                EW_troubled_faces[0] == 1,
                EW_fluxes_2nd_order[0],
                self.EW_fluxes,
            )

    def dont_revise_solution(self, u0: np.ndarray, dt: float):
        return

    def get_dynamics(self) -> np.ndarray:
        """
        dudt + d(au)dx + d(bu)dy = 0
        returns:
            dudt    (ny, nx)
        """
        return -self.hx_recip * (
            self.EW_fluxes[:, 1:] - self.EW_fluxes[:, :-1]
        ) + -self.hy_recip * (self.NS_fluxes[1:, :] - self.NS_fluxes[:-1, :])

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

    def periodic_error(self, norm: str = "l1"):
        """
        args:
            norm    'l1', 'l2', or 'linf'
        returns:
            norm specified error    (ny, nx)
        """
        approx = self.u[-1]
        truth = self.u[0]
        if norm == "l1":
            return np.sum(np.abs(approx - truth) * self.hx * self.hy)
        if norm == "l2":
            return np.sqrt(np.sum(np.power(approx - truth, 2)) * self.hx * self.hy)
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
        im1 = axs[0, 0].imshow(data1, cmap="cool", vmin=hmin, vmax=hmax, extent=bounds)
        axs[0, 1].imshow(data2, cmap="cool", vmin=hmin, vmax=hmax, extent=bounds)
        axs[0, 2].imshow(data3, cmap="cool", vmin=hmin, vmax=hmax, extent=bounds)
        axs[1, 0].imshow(data4, cmap="cool", vmin=hmin, vmax=hmax, extent=bounds)
        axs[1, 1].imshow(data5, cmap="cool", vmin=hmin, vmax=hmax, extent=bounds)
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
        axs[1, 2].set_title(f"u(t = {self.t[-1]:.3f}) - u(t = {self.t[0]:.3f})")

        # Add a title to the entire figure
        fig.suptitle("Heatmap Grid")

        # Show the plot
        plt.show()
