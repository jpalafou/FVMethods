# advection solution schems
import numpy as np
import os
import pickle
from typing import Tuple, Dict
from finite_volume.initial_conditions import generate_ic
from finite_volume.integrate import Integrator
from finite_volume.fvscheme import ConservativeInterpolation, TransverseIntegral
from finite_volume.mathematiques import gauss_lobatto
from finite_volume.sed import compute_alpha_1d, compute_alpha_2d
from finite_volume.a_priori import mpp_limiter
from finite_volume.a_posteriori import (
    minmod,
    moncen,
    find_trouble,
    compute_MUSCL_interpolations_1d,
    compute_MUSCL_interpolations_2d,
    compute_PP2D_interpolations,
    broadcast_troubled_cells_to_faces_1d,
    broadcast_troubled_cells_to_faces_2d,
    broadcast_troubled_cells_to_faces_with_blending_1d,
    broadcast_troubled_cells_to_faces_with_blending_2d,
)
from finite_volume.utils import (
    rk4_dt_adjust,
    batch_convolve2d,
)
from finite_volume.riemann import upwinding


# class definition
class AdvectionSolver(Integrator):
    """
    args:
        u0                          initial condition, keywork or callable function
        bc                          "periodic" or "dirichlet"
        const                       for dirichlet bc
        n                           tuple of number of cells in x and y
        x                           tuple of boundaries in x
        y                           tuple of boundaries in y
        t0                          starting time
        snapshot_dt                 dt for snapshots
        num_snapshots               number of times to evolve system by snapshot_dt
        v                           tuple of floating point velocity components or
                                    callable function of x and y
        courant                     stability condition
        order                       accuracy requirement for polynomial interpolation
        flux_strategy               'gauss-legendre' or 'transverse'
    slope limiter settings - - - - -
    a priori limiting ~
        apriori_limiting            whether to follow zhang and shu mpp limiting
        mpp_lite                    cell center is the only interior point
    a posteriori limiting ~
        aposteriori_limiting        whether to call trouble detection and 2d fallback
        fallback_limiter            'moncen', 'minmod', or 'PP2D'
        convex                      a more mpp version of a posteriori limiting
        hancock                     predictor corrector scheme for fallback
        fallback_to_first_order     fallback again to first order in the fallback scheme
        cause_trouble               set all cells to be troubled, forcing 2d fallback
    ~
        SED                         whether to enable smooth extrema detection
        NAD                         simulation NAD tolerance for a posteriori limiting
                                        set to None or +inf to disable NAD
        PAD                         physical admissibility detection (lower, upper)
                                        set to None or (-inf, +inf) to disable PAD
        visualization_tolerance     tolerance for whether to visualize theta and/or
                                        troubled cells based on some violation
                                        set to None or -inf to visualize simulation
                                        values
    - - - - -
        adjust_time_step            whether to reduce timestep for order >4
        modify_time_step            whether to conditionally reduce dt by half
        mpp_tolerance               maximum principle tolerance for adaptive time step
        progress_bar                whether to print a progress bar in the loop
        load                        whether to load precalculated solution
        save                        whether to overwrite saved instance
        save_directory              directory from which to read/write
    returns:
        u   array of saved states
    """

    def __init__(
        self,
        u0: str = "square",
        bc: str = "periodic",
        const: float = None,
        n: tuple = (32, 32),
        x: tuple = (0, 1),
        y: tuple = None,
        t0: float = 0,
        snapshot_dt: float = 1.0,
        num_snapshots: int = 1,
        v: tuple = (2, 1),
        courant: float = 0.8,
        order: int = 1,
        flux_strategy: str = "gauss-legendre",
        apriori_limiting: bool = False,
        mpp_lite: bool = False,
        aposteriori_limiting: bool = False,
        fallback_limiter: str = "moncen",
        convex: bool = False,
        hancock: bool = False,
        fallback_to_first_order: bool = False,
        cause_trouble: bool = False,
        SED: bool = False,
        NAD: float = 1e-10,
        PAD: tuple = (0, 1),
        visualization_tolerance: float = None,
        adjust_time_step: bool = False,
        modify_time_step: bool = False,
        mpp_tolerance: float = 1e-10,
        progress_bar: bool = True,
        load: bool = True,
        save: bool = True,
        save_directory: str = "data/solutions/",
    ):
        # create filename out of the initialization arguments
        self.load = load
        u0_str = u0.__name__ if callable(u0) else str(u0)
        v_str = v.__name__ if callable(v) else str(v)
        filename_components = [
            u0_str,
            bc,
            const,
            n,
            x,
            y,
            t0,
            snapshot_dt,
            num_snapshots,
            v_str,
            courant,
            order,
            flux_strategy,
            apriori_limiting,
            mpp_lite,
            aposteriori_limiting,
            fallback_limiter,
            convex,
            hancock,
            fallback_to_first_order,
            cause_trouble,
            SED,
            NAD,
            PAD,
            visualization_tolerance,
            adjust_time_step,
            modify_time_step,
            mpp_tolerance,
            progress_bar,
        ]
        self._filename = "_".join(str(component) for component in filename_components)
        self._save_directory = save_directory
        self.save = save

        # dimensionality
        self.ndim = 1 if isinstance(n, int) else 2

        # spatial discretization in x
        if self.ndim == 1:
            nx = n
        elif self.ndim == 2 and len(n) < 2:
            nx = n[0]
        else:
            nx = n[1]
        self.x_interface = np.linspace(x[0], x[1], num=nx + 1)  # cell interfaces
        self.x = 0.5 * (self.x_interface[:-1] + self.x_interface[1:])  # cell centers
        self.Lx = x[1] - x[0]  # domain size in x
        self.hx = self.Lx / nx  # cell size in x
        self.hx_recip = nx / self.Lx  # 1 / h

        # spatial discretization in y
        if self.ndim == 1:
            self.y = None
            ny = 1
            self.hy = 1
            self.Ly = 1
            self.hy_recip = 1
            self.n_cells = nx
        if self.ndim == 2:
            ny = n[0]
            y = x if y is None else y
            self.y_interface = np.linspace(y[0], y[1], num=ny + 1)  # cell interfaces
            self.y = 0.5 * (
                self.y_interface[:-1] + self.y_interface[1:]
            )  # cell centers
            self.Ly = y[1] - y[0]  # domain size in y
            self.hy = self.Ly / ny  # cell size in y
            self.hy_recip = ny / self.Ly  # 1 / h
            self.n_cells = nx * ny

        # maximum expected advection velocities
        if self.ndim == 1:  # uniform 1d velocity
            if isinstance(v, tuple) or isinstance(v, list):
                raise BaseException("Defined vector velocity for 1d field")
            vx_max, vy_max = abs(v), 0
        if self.ndim == 2:
            if isinstance(v, int):
                raise BaseException("Defined scalar velocity for 2d field")
            if isinstance(v, tuple) or isinstance(v, list):  # uniform 2d velocity
                if len(v) < 2:
                    vx_max, vy_max = abs(v[0]), abs(v[0])
                else:
                    vx_max, vy_max = abs(v[0]), abs(v[1])
            if callable(v):  # non-uniform 2d velocity
                # precompute velocity at cell corners and use this as estimate for max
                xx, yy = np.meshgrid(self.x_interface, self.y_interface)
                v_max = v(xx, yy)
                vx_max, vy_max = np.max(abs(v_max[0])), np.max(abs(v_max[1]))

        # time discretization
        self.adjust_time_step = adjust_time_step
        self.order = order
        self.p = order - 1
        v_over_h = vx_max / self.hx + vy_max / self.hy
        if v_over_h == 0:
            print("0 velocity case: setting v / h to 0.1")
            v_over_h = 0.1
        self.courant = courant
        if adjust_time_step and order > 4:
            adjusted_courant = min(rk4_dt_adjust(nx, order), rk4_dt_adjust(ny, order))
            if adjusted_courant < courant:
                self.courant = adjusted_courant
                print(f"Reassigned C={self.courant} for order {order}")
        dt = self.courant / v_over_h

        # initial condition
        if isinstance(u0, str):
            u0 = generate_ic(type=u0, x=self.x, y=self.y)
        if callable(u0):
            if self.ndim == 1:
                u0 = u0(x=self.x)
            if self.ndim == 2:
                u0 = u0(x=self.x, y=self.y)

        # initialize simulation/visualized theta and trouble
        self.ones_i = np.ones_like(u0, dtype="int")
        self.ones_f = np.ones_like(u0, dtype="float")
        self.theta = self.ones_f
        self.visualized_theta = self.ones_f
        self.trouble = 0 * self.ones_i
        self.visualized_trouble = 0 * self.ones_i

        # initialize timeseries lists
        self.u0_min = np.min(u0)
        self.u0_max = np.max(u0)
        self.all_simulation_times = [t0]
        self.min_history = [self.u0_min]
        self.max_history = [self.u0_max]

        # initialize snapshots
        self.u_snapshots = [(t0, u0)]
        self.theta_snapshots = [(t0, self.theta)]
        self.visualized_theta_snapshots = [(t0, self.visualized_theta)]
        self.trouble_snapshots = [(t0, self.trouble)]
        self.visualized_trouble_snapshots = [(t0, self.visualized_trouble)]

        # initialize integrator
        self.timestamps = [t0]
        super().__init__(
            u0=u0,
            dt=dt,
            snapshot_dt=snapshot_dt,
            num_snapshots=num_snapshots,
            t0=t0,
            progress_bar=progress_bar,
        )

        # boundary conditions
        self.const = const
        constants = dict(
            u=self.const,
            trouble=0,
        )
        if bc == "periodic":
            self.bc_config = dict(mode="wrap")
            self.constant_bc_config = {key: {} for key, _ in constants.items()}
        if bc == "dirichlet":
            self.bc_config = dict(mode="constant")
            self.constant_bc_config = {
                key: dict(constant_values=item) for key, item in constants.items()
            }

        # initialize limiting
        self.apriori_limiting = apriori_limiting
        self.mpp_lite = mpp_lite
        self.aposteriori_limiting = aposteriori_limiting
        if fallback_limiter == "minmod":
            self.fallback_limiter = minmod
        elif fallback_limiter == "moncen":
            self.fallback_limiter = moncen
        elif fallback_limiter == "PP2D":
            if self.ndim != 2:
                raise BaseException("PP2D limiting is for two-dimensional solutions.")
            if fallback_to_first_order:
                raise BaseException("PP2D does not fall back to first order.")
            self.fallback_limiter = compute_PP2D_interpolations
        self.SED = SED

        self.convex = convex
        self.hancock = hancock
        self.fallback_to_first_order = fallback_to_first_order

        # arrays for storing local min/max
        self.m = 0 * self.ones_f
        self.M = 0 * self.ones_f

        # initialize cause_trouble and udot_evaluation_count
        self.cause_trouble = cause_trouble
        self.udot_evaluation_count = 0
        # initialize tolerances
        self.NAD = np.inf if NAD is None else NAD
        self.visualization_tolerance = (
            -np.inf
            if visualization_tolerance is None or cause_trouble
            else visualization_tolerance
        )
        # initialize PAD
        self.PAD = (-np.inf, np.inf) if PAD is None else PAD
        self.mpp_tolerance = np.inf if mpp_tolerance is None else mpp_tolerance
        self.approximated_maximum_principle = (
            self.PAD[0] - self.mpp_tolerance,
            self.PAD[1] + self.mpp_tolerance,
        )

        # quadrature setup
        self.flux_strategy = flux_strategy if self.ndim > 1 else "gauss-legendre"
        cons_left = ConservativeInterpolation.construct_from_order(
            self.p + 1, "left"
        ).nparray()
        conservative_stencil_size = len(cons_left)
        self.conservative_width = conservative_stencil_size // 2
        self.transverse_width = 0

        # interpolate line averages
        line_interpolation_positions = []
        if self.flux_strategy == "gauss-legendre":
            # legendre gauss quadrature
            n_legendre_gauss = self.p // 2 + 1
            assert 2 * n_legendre_gauss - 1 >= self.p
            leg_points, leg_weights = np.polynomial.legendre.leggauss(n_legendre_gauss)
            leg_points /= 2
            leg_weights /= 2
            self.leg_weights = np.array(leg_weights).reshape(-1, 1, 1)
            line_interpolation_positions = list(leg_points)
        elif self.flux_strategy == "transverse":
            line_interpolation_positions = ["center"]
        list_of_line_stencils = []
        for x in line_interpolation_positions:
            s = ConservativeInterpolation.construct_from_order(self.p + 1, x).nparray(
                size=conservative_stencil_size
            )
            list_of_line_stencils.append(s)
        self.line_stencils = np.array(list_of_line_stencils)

        # interpolate pointwise averages
        if self.apriori_limiting and not self.mpp_lite:
            n_lobatto_gauss = self.p // 2 + 2
            assert 2 * n_lobatto_gauss - 3 >= self.p
            if n_lobatto_gauss > 2:
                lob_points, lob_weights = gauss_lobatto(n_lobatto_gauss)
                lob_points /= 2
                lob_weights /= 2
                interior_lob_points = lob_points[1:-1]
                # warn use if C is too high for MPP
                C_mpp = min(lob_weights)
                self.dt_min = C_mpp / v_over_h
                # check if timestep is small enough for mpp
                if self.dt * v_over_h > C_mpp:
                    print(
                        "WARNING: Maximum principle preserving not satisfied.\nTry a ",
                        f"CFL factor less than {C_mpp:.5f}\n",
                    )
            else:
                interior_lob_points = []
            point_interpolation_positions = (
                ["left"] + list(interior_lob_points) + ["right"]
            )
        else:
            point_interpolation_positions = ["left", "right"]
        list_of_point_stencils = []
        for x in point_interpolation_positions:
            s = ConservativeInterpolation.construct_from_order(self.p + 1, x).nparray(
                size=conservative_stencil_size
            )
            list_of_point_stencils.append(s)
        self.pointwise_stencils = np.array(list_of_point_stencils)

        if self.flux_strategy == "transverse":
            leg_points = np.array([0.0])
            self.transverse_stencil = TransverseIntegral.construct_from_order(
                self.p + 1
            ).nparray()
            self.transverse_width = len(self.transverse_stencil) // 2

        if self.mpp_lite:
            self.cell_center_stencil = ConservativeInterpolation.construct_from_order(
                self.p + 1, "center"
            ).nparray(size=conservative_stencil_size)

        self.n_line_stencils = self.line_stencils.shape[0]
        self.n_pointwise_stencils = self.pointwise_stencils.shape[0]

        # ghost width
        self.riemann_gw = max(self.transverse_width, 1)
        self.excess_riemann_gw = self.riemann_gw - 1
        self.excess_transverse_gw = self.riemann_gw - self.transverse_width

        # velocity at cell interfaces
        if self.ndim == 1:
            self.a = v * np.ones(nx + 1)
            self.a_cell_centers = v * np.ones(nx)
        if self.ndim == 2:
            # x and y values at interface points
            na = np.newaxis
            # EW interface
            EW_interface_x, EW_interface_y = np.meshgrid(self.x_interface, self.y)
            EW_interface_x = np.repeat(EW_interface_x[na], len(leg_points), axis=0)
            EW_interface_y = np.repeat(EW_interface_y[na], len(leg_points), axis=0)
            EW_interface_y += leg_points[:, na, na] * self.hy
            EW_midpoint_x, EW_midpoint_y = np.meshgrid(self.x_interface, self.y)
            # NS interface
            NS_interface_x, NS_interface_y = np.meshgrid(self.x, self.y_interface)
            NS_interface_x = np.repeat(NS_interface_x[na], len(leg_points), axis=0)
            NS_interface_y = np.repeat(NS_interface_y[na], len(leg_points), axis=0)
            NS_interface_x += leg_points[:, na, na] * self.hx
            NS_midpoint_x, NS_midpoint_y = np.meshgrid(self.x, self.y_interface)
            # evaluate v components normal to cell interfaces
            xx_center, yy_center = np.meshgrid(self.x, self.y)
            if isinstance(v, tuple):
                self.a = v[0] * np.ones(EW_interface_y.shape)
                self.a_midpoint = v[0] * np.ones(EW_midpoint_x.shape)
                self.a_cell_centers = v[0] * np.ones(xx_center.shape)
                if len(v) == 1:
                    self.b = v[0] * np.ones(NS_interface_x.shape)
                    self.b_midpoint = v[0] * np.ones(NS_midpoint_y.shape)
                    self.b_cell_centers = v[0] * np.ones(yy_center.shape)
                elif len(v) == 2:
                    self.b = v[1] * np.ones(NS_interface_x.shape)
                    self.b_midpoint = v[1] * np.ones(NS_midpoint_y.shape)
                    self.b_cell_centers = v[1] * np.ones(yy_center.shape)
            if callable(v):
                self.a = v(EW_interface_x, EW_interface_y)[0]
                self.b = v(NS_interface_x, NS_interface_y)[1]
                self.a_midpoint = v(EW_midpoint_x, EW_midpoint_y)[0]
                self.b_midpoint = v(NS_midpoint_x, NS_midpoint_y)[1]
                self.a_cell_centers = v(xx_center, yy_center)[0]
                self.b_cell_centers = v(xx_center, yy_center)[1]
            if self.flux_strategy == "transverse":  # apply boundary for transverse flux
                self.a = self.apply_bc(
                    self.a, gw=[(0,), (self.transverse_width,), (0,)]
                )
                self.b = self.apply_bc(
                    self.b, gw=[(0,), (0,), (self.transverse_width,)]
                )

        # fluxes
        if self.ndim == 1:
            self.f = np.zeros(nx + 1)
            self.g = np.zeros(nx + 1)
        if self.ndim == 2:
            self.f = np.zeros((ny, nx + 1))  # east/west fluxes
            self.g = np.zeros((ny + 1, nx))  # north/south fluxes

        # dynamic function assignment
        self.riemann_solver = upwinding

        if self.ndim == 1:
            self.get_fluxes = self.get_fluxes_1d
            self.compute_alpha = compute_alpha_1d
        if self.ndim == 2:
            self.get_fluxes = self.get_fluxes_2d
            self.compute_alpha = compute_alpha_2d

        if modify_time_step:
            self.looks_good = self.check_mpp

        if self.flux_strategy == "gauss-legendre":
            self.integrate_fluxes = self.gauss_legendre_integral
        elif self.flux_strategy == "transverse":
            self.integrate_fluxes = self.transverse_integral

        if self.aposteriori_limiting:
            if self.ndim == 1:
                self.aposteriori_limiter = self.revise_fluxes_1d
            elif self.ndim == 2:
                self.aposteriori_limiter = self.revise_fluxes_2d

    def get_dynamics(self) -> np.ndarray:
        """
        advection equation:
        dudt + dfdx + dgdy = 0
        f and g are fluxes in x and y
        """
        dfdx = self.hx_recip * (self.f[..., 1:] - self.f[..., :-1])
        dgdy = self.hy_recip * (self.g[1:, ...] - self.g[:-1, ...])
        return -dfdx + -dgdy

    def udot(self, u: np.ndarray, t: float = None, dt: float = None) -> np.ndarray:
        """
        args:
            u       (ny, nx)
            t_i     time at which u is defined
            dt      timestep size from t_i to t_i+1
        returns:
            dudt    (ny, nx) at t_i
        """
        self.udot_evaluation_count += 1
        if not (self.cause_trouble and self.aposteriori_limiting):
            self.get_fluxes(u=u)  # high order flux update
        if self.aposteriori_limiting:
            self.aposteriori_limiter(u=u, dt=dt)
        return self.get_dynamics()

    def apply_bc(
        self,
        u_without_ghost_cells: np.ndarray,
        gw: {tuple, list, int},
        mode="u",
    ) -> np.ndarray:
        """
        args:
            u_without_ghost_cells   (ny, nx)
            gw  a: int
                applies a ghost width of a around the entire array
                (a, b): tuple
                applies a left hand ghost width of a and a right hand ghost with of b
                around the entire array
                [(a, b), (c, d), (e, f), ...]: list of tuples for every dimension
                tip:
                use [(a,), (c,), (e,), ...] for symmetric boundaries
            param   'u', 'trouble'
        returns:
            u with periodic ghost zones
        """
        return np.pad(
            u_without_ghost_cells,
            pad_width=gw,
            **self.bc_config,
            **self.constant_bc_config[mode],
        )

    def get_fluxes_1d(self, u: np.ndarray):
        """
        args:
            u   (nx,)
        overwrites:
            self.a  (nx + 1,)
        """
        # interpolate points from line averages
        points = batch_convolve2d(
            arr=self.apply_bc(u, gw=self.conservative_width + 1).reshape(1, -1),
            kernel=self.pointwise_stencils.reshape(self.n_pointwise_stencils, 1, -1),
        )
        points = points[0, :, 0, :]  # (# of interpolated points, # cells x
        # slope limiting
        mpp_limiting_points = points
        if self.mpp_lite:
            midpoints = self.compute_cell_center(u)
            mpp_limiting_points = np.concatenate((mpp_limiting_points, midpoints))
        theta, M_i, m_i = mpp_limiter(
            u=self.apply_bc(u, gw=2),
            points=mpp_limiting_points,
            ones=not self.apriori_limiting,
            zeros=self.cause_trouble,
        )
        # PAD
        PAD = np.logical_or(
            m_i < self.approximated_maximum_principle[0],
            M_i > self.approximated_maximum_principle[1],
        )
        # smooth extrema detection
        alpha = self.compute_alpha(self.apply_bc(u, gw=4), zeros=not self.SED)
        not_smooth_extrema = alpha < 1
        # PAD then SED
        theta = np.where(PAD, theta, np.where(not_smooth_extrema, theta, 1.0))
        # limit slopes
        first_order_fallback = self.apply_bc(u, gw=1)
        points = theta * (points - first_order_fallback) + first_order_fallback
        # riemann problem
        right_points = points[-1, :-1]
        left_points = points[0, 1:]
        self.f[...] = self.riemann_solver(
            v=self.a, left_value=right_points, right_value=left_points
        )

    def get_fluxes_2d(self, u: np.ndarray):
        """
        args:
            u   (ny, nx)
        overwrites:
            self.a  (ny, nx + 1)
            self.b  (ny + 1, nx)
        """
        # interpolate line averages from cell averages
        horizontal_lines = batch_convolve2d(
            arr=self.apply_bc(u, gw=self.conservative_width + self.riemann_gw)[
                np.newaxis
            ],
            kernel=self.line_stencils.reshape(self.n_line_stencils, -1, 1),
        )
        vertical_lines = batch_convolve2d(
            arr=self.apply_bc(u, gw=self.conservative_width + self.riemann_gw)[
                np.newaxis
            ],
            kernel=self.line_stencils.reshape(self.n_line_stencils, 1, -1),
        )
        # interpolate points from line averages
        horizontal_points = batch_convolve2d(
            arr=horizontal_lines[0],
            kernel=self.pointwise_stencils.reshape(self.n_pointwise_stencils, 1, -1),
        )
        vertical_points = batch_convolve2d(
            arr=vertical_lines[0],
            kernel=self.pointwise_stencils.reshape(self.n_pointwise_stencils, -1, 1),
        )
        # slope limiting
        h_shp = horizontal_points.shape
        hps = horizontal_points.reshape(h_shp[0] * h_shp[1], h_shp[2], h_shp[3])
        v_shp = vertical_points.shape
        vps = vertical_points.reshape(v_shp[0] * v_shp[1], v_shp[2], v_shp[3])
        mpp_limiting_points = np.concatenate((hps, vps))
        if self.mpp_lite:
            midpoints = self.compute_cell_center(u)
            mpp_limiting_points = np.concatenate((mpp_limiting_points, midpoints))
        theta, M_ij, m_ij = mpp_limiter(
            u=self.apply_bc(u, gw=self.riemann_gw + 1),
            points=mpp_limiting_points,
            ones=not self.apriori_limiting,
            zeros=self.cause_trouble,
        )
        # PAD
        PAD = np.logical_or(
            m_ij < self.approximated_maximum_principle[0],
            M_ij > self.approximated_maximum_principle[1],
        )
        # smooth extrema detection
        alpha = self.compute_alpha(
            self.apply_bc(u, gw=self.riemann_gw + 3), zeros=not self.SED
        )
        not_smooth_extrema = alpha < 1
        # PAD then SED
        theta = np.where(PAD, theta, np.where(not_smooth_extrema, theta, 1.0))
        # limit slopes
        fallback = self.apply_bc(u, gw=self.riemann_gw)
        horizontal_points = theta * (horizontal_points - fallback) + fallback
        vertical_points = theta * (vertical_points - fallback) + fallback
        # remove excess due to uniform ghost zone
        trans_slice = slice(
            self.excess_transverse_gw or None, -self.excess_transverse_gw or None
        )
        riemann_slice = slice(
            self.excess_riemann_gw or None, -self.excess_riemann_gw or None
        )
        horizontal_points = horizontal_points[:, :, trans_slice, riemann_slice]
        vertical_points = vertical_points[:, :, riemann_slice, trans_slice]
        ns_pointwise_fluxes = self.riemann_solver(
            v=self.b,
            left_value=vertical_points[:, -1, :-1, :],  # north points
            right_value=vertical_points[:, 0, 1:, :],  # south points
        )
        ew_pointwise_fluxes = self.riemann_solver(
            v=self.a,
            left_value=horizontal_points[:, -1, :, :-1],  # east points
            right_value=horizontal_points[:, 0, :, 1:],  # west points
        )
        # integrate fluxes with gauss-legendre quadrature
        self.f[...] = self.integrate_fluxes(ew_pointwise_fluxes, axis=0)
        self.g[...] = self.integrate_fluxes(ns_pointwise_fluxes, axis=1)

    def compute_cell_center(self, u: np.ndarray) -> np.ndarray:
        """
        args:
            u       cell volume averages with padding (m + p,) or (m + p, n + q)
        returns:
            out     cell midpoints  (m,) or (m, n)
        """
        if self.ndim == 1:
            midpoints = batch_convolve2d(
                arr=self.apply_bc(u, gw=self.conservative_width + 1)[np.newaxis],
                kernel=self.cell_center_stencil.reshape(1, -1),
            )
            return midpoints[0, 0, ...]
        elif self.ndim == 2:
            horizontal_lines = batch_convolve2d(
                arr=self.apply_bc(u, gw=self.conservative_width + self.riemann_gw)[
                    np.newaxis
                ],
                kernel=self.cell_center_stencil.reshape(1, -1, 1),
            )
            midpoints = batch_convolve2d(
                arr=horizontal_lines[0],
                kernel=self.cell_center_stencil.reshape(1, 1, -1),
            )
            return midpoints[0, ...]

    def find_trouble(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        args:
            u           cell volume averages    (m,) or (m, n)
            dt          time step size
        returns:
            trouble     boolean array           (m,) or (m, n)
        """
        unew = u + dt * self.get_dynamics()
        trouble = find_trouble(
            u=self.apply_bc(u, gw=1),
            u_candidate=self.apply_bc(unew, gw=3),
            NAD=self.NAD,
            PAD=self.approximated_maximum_principle,
            SED=self.SED,
            ones=self.cause_trouble,
        )
        return trouble

    def revise_fluxes_1d(self, u: np.ndarray, dt: float):
        """
        args:
            u           cell volume averages    (m,) or (m, n)
            dt          time step size
        overwrites:
            self.f
        """
        # compute fallback fluxes
        left_fallback_face, right_fallback_face = compute_MUSCL_interpolations_1d(
            u=self.apply_bc(u, gw=1),
            slope_limiter=self.fallback_limiter,
            fallback_to_1st_order=self.fallback_to_first_order,
            PAD=self.approximated_maximum_principle,
            hancock=self.hancock,
            dt=dt,
            h=self.hx,
            v_cell_centers=self.a_cell_centers,
        )
        left_fallback_face = self.apply_bc(left_fallback_face, gw=1)
        right_fallback_face = self.apply_bc(right_fallback_face, gw=1)
        fallback_fluxes = self.riemann_solver(
            v=self.a,
            left_value=right_fallback_face[:-1],
            right_value=left_fallback_face[1:],
        )
        # find troubled cells
        trouble = self.find_trouble(u, dt)
        # revise fluxes
        if not self.convex:
            troubled_interface_mask = broadcast_troubled_cells_to_faces_1d(trouble)
        else:
            troubled_interface_mask = (
                broadcast_troubled_cells_to_faces_with_blending_1d(
                    self.apply_bc(trouble, gw=2, mode="trouble")
                )
            )
        self.f[...] = (
            1 - troubled_interface_mask
        ) * self.f + troubled_interface_mask * fallback_fluxes

    def revise_fluxes_2d(self, u: np.ndarray, dt: float):
        """
        args:
            u           cell volume averages    (m,) or (m, n)
            dt          time step size
        overwrites:
            self.f
            self.g
        """
        # compute fallback fluxes
        if self.fallback_limiter.__name__ in {"minmod", "moncen"}:
            fallback_faces = compute_MUSCL_interpolations_2d(
                u=self.apply_bc(u, gw=1),
                slope_limiter=self.fallback_limiter,
                fallback_to_1st_order=self.fallback_to_first_order,
                PAD=self.PAD,
                hancock=self.hancock,
                dt=self.dt,
                h=(1 / self.hx, 1 / self.hy),
                v_cell_centers=self.a_cell_centers,
            )
        elif self.fallback_limiter.__name__ == "compute_PP2D_interpolations":
            fallback_faces = self.fallback_limiter(
                u=self.apply_bc(u, gw=1),
                hancock=self.hancock,
                dt=self.dt,
                h=(1 / self.hx, 1 / self.hy),
                v_cell_centers=self.a_cell_centers,
            )
        north_face = self.apply_bc(fallback_faces[1][1], gw=1)
        south_face = self.apply_bc(fallback_faces[1][0], gw=1)
        east_face = self.apply_bc(fallback_faces[0][1], gw=1)
        west_face = self.apply_bc(fallback_faces[0][0], gw=1)
        fallback_fluxes_x = self.riemann_solver(
            v=self.a_midpoint,
            left_value=east_face[1:-1, :-1],
            right_value=west_face[1:-1, 1:],
        )
        fallback_fluxes_y = self.riemann_solver(
            v=self.b_midpoint,
            left_value=south_face[:-1, 1:-1],
            right_value=north_face[1:, 1:-1],
        )
        # find troubled cells
        trouble = self.find_trouble(u, dt)
        # overwrite fluxes
        if not self.convex:
            (
                troubled_interface_mask_x,
                troubled_interface_mask_y,
            ) = broadcast_troubled_cells_to_faces_2d(trouble)
        else:
            (
                troubled_interface_mask_x,
                troubled_interface_mask_y,
            ) = broadcast_troubled_cells_to_faces_with_blending_2d(
                self.apply_bc(trouble, gw=2, mode="trouble")
            )
        self.f[...] = (
            1 - troubled_interface_mask_x
        ) * self.f + troubled_interface_mask_x * fallback_fluxes_x
        self.g[...] = (
            1 - troubled_interface_mask_y
        ) * self.g + troubled_interface_mask_y * fallback_fluxes_y

    def gauss_legendre_integral(self, pointwise_fluxes: np.ndarray, axis: int):
        """
        args:
            pointwise_fluxes    (p, m, n)
        returns:
            out                 (1, 1, m, n)
        """
        na = np.newaxis
        return np.sum(pointwise_fluxes * self.leg_weights, axis=0)[na, na]

    def transverse_integral(self, pointwise_fluxes: np.ndarray, axis: int):
        """
        args:
            pointwise_fluxes    (1, m + excess, n) or (1, m, n + excess)
        returns:
            out                 (1, 1, m, n)
        """
        kernel_shape = [1, 1, 1]
        kernel_shape[axis + 1] = len(self.transverse_stencil)
        return batch_convolve2d(
            arr=pointwise_fluxes, kernel=self.transverse_stencil.reshape(kernel_shape)
        )

    def rkorder(self, ssp: bool = True):
        """
        rk integrate to an order that matches the spatial order
        """
        if self.order < 2:
            self.euler()
        elif self.order < 3:
            if ssp:
                self.ssprk2()
            else:
                self.rk2()
        elif self.order < 4:
            if ssp:
                self.ssprk3()
            else:
                self.rk3()
        else:
            self.rk4()

    def periodic_error(self, norm: str = "l1"):
        """
        args:
            norm    'l1', 'l2', or 'linf'
        returns:
            norm specified error    (ny, nx)
        """
        approx = self.u_snapshots[-1][1]
        truth = self.u_snapshots[0][1]
        if norm == "l1":
            return np.sum(np.abs(approx - truth) * self.hx * self.hy)
        if norm == "l2":
            return np.sqrt(np.sum(np.power(approx - truth, 2)) * self.hx * self.hy)
        if norm == "inf":
            return np.max(np.abs(approx - truth))

    def check_mpp(self, u):
        return not np.logical_or(
            np.any(u < self.approximated_maximum_principle[0]),
            np.any(u > self.approximated_maximum_principle[1]),
        )

    def append_to_timeseries_lists(self):
        self.all_simulation_times.append(self.t0)
        self.min_history.append(np.min(self.u0))
        self.max_history.append(np.max(self.u0))

    def snapshot(self):
        snap_time = self.t0
        self.timestamps.append(snap_time)
        self.u_snapshots.append((snap_time, self.u0))
        self.theta_snapshots.append(
            (snap_time, self.theta / self.udot_evaluation_count)
        )
        self.visualized_theta_snapshots.append(
            (snap_time, self.visualized_theta / self.udot_evaluation_count)
        )
        self.trouble_snapshots.append(
            (snap_time, self.trouble / self.udot_evaluation_count)
        )
        self.visualized_trouble_snapshots.append(
            (snap_time, self.visualized_trouble / self.udot_evaluation_count)
        )

    def step_cleanup(self):
        """
        self.t0 has been overwritten with t0 + dt
        self.u0 has been overwritten with the state at t0 + dt
        """
        # clear theta sum, troubled cell sum, and evaluation count
        self.theta[...] = 0.0
        self.visualized_theta[...] = 0
        self.trouble[...] = 0
        self.visualized_trouble[...] = 0
        self.udot_evaluation_count = 0
        self.append_to_timeseries_lists()

    def refine_timestep(self, dt):
        return dt / 2

    def pre_integrate(self, method_name):
        # find filepath where solution is/will be stored
        self._filename = self._filename + "_" + method_name + ".pkl"
        self.filepath = self._save_directory + self._filename
        # load the solution if it already exists
        if os.path.isfile(self.filepath) and self.load:
            with open(self.filepath, "rb") as thisfile:
                loaded_instance = pickle.load(thisfile)
                attribute_names = [
                    attr
                    for attr in dir(loaded_instance)
                    if not callable(getattr(loaded_instance, attr))
                    and not attr.startswith("_")
                ]
                for attribute in attribute_names:
                    value = getattr(loaded_instance, attribute)
                    setattr(self, attribute, value)
            return False
        # otherwise proceed to integration
        try:
            os.makedirs(self._save_directory)
        except OSError:
            pass
        print("New solution instance...")
        return True

    def write_to_file(self):
        # Save the instance to a file
        if self.save:
            with open(self.filepath, "wb") as thisfile:
                pickle.dump(self, thisfile)
            print(f"Wrote a solution up to t = {self.t0} located at {self.filepath}\n")

    def post_integrate(self):
        """
        only runs if new solution is computed
        """
        # compute max and min history
        min_history = np.asarray(self.min_history)
        max_history = np.asarray(self.max_history)
        self.mean_min = np.mean(min_history)
        self.std_min = np.std(min_history)
        self.abs_min = np.min(min_history)
        self.mean_max = np.mean(max_history)
        self.std_max = np.std(max_history)
        self.abs_max = np.max(max_history)
        self.write_to_file()

    def minmax(self):
        headers = [
            "abs min/max",
            "mean min/max",
            "std min/max",
        ]
        values1 = [
            self.abs_min,
            self.mean_min,
            self.std_min,
        ]
        values2 = [
            self.abs_max,
            self.mean_max,
            self.std_max,
        ]
        print("---------------------------------")
        print("{:>14}{:>14}     {:>11}".format(*headers))
        print("{:14.5e}{:14.5e} +/- {:11.5e}".format(*values1))
        print("{:14.5e}{:14.5e} +/- {:11.5e}".format(*values2))
        print()
        print("{:>10}{:10.5f}".format("time (s):", self.solution_time))
        print()

    def compute_violations(self) -> Tuple[np.ndarray, Dict]:
        mins = np.array(self.min_history)
        maxs = np.array(self.max_history)
        lower_bound_violations = mins - self.PAD[0]
        upper_bound_violations = self.PAD[1] - maxs
        violations = np.minimum(lower_bound_violations, upper_bound_violations)
        # negative values indicate violation
        stats = {}
        stats["worst"] = np.min(violations)
        stats["mean"] = np.mean(violations)
        stats["standard deviation"] = np.std(violations)
        stats["lower bound violation frequency"] = np.sum(
            np.where(lower_bound_violations < 0, 1, 0)
        ) / (self.step_count + 1)
        stats["upper bound violation frequency"] = np.sum(
            np.where(upper_bound_violations < 0, 1, 0)
        ) / (self.step_count + 1)
        stats["violation frequency"] = np.sum(np.where(violations < 0, 1, 0)) / (
            self.step_count + 1
        )
        return violations, stats
