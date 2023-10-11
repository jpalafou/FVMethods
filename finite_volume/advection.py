# advection solution schems
import numpy as np
import os
import pickle
from finite_volume.initial_conditions import generate_ic
from finite_volume.integrate import Integrator
from finite_volume.fvscheme import ConservativeInterpolation, TransverseIntegral
from finite_volume.mathematiques import gauss_lobatto
from finite_volume.trouble_detection import (
    detect_smooth_extrema,
    detect_no_smooth_extrema,
    compute_fallback_faces,
)
from finite_volume.utils import (
    rk4_dt_adjust,
    stack,
    apply_stencil,
    chop,
    f_of_3_neighbors,
    f_of_4_neighbors,
)


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
        apriori_limiting            whether to follow zhang and shu mpp limiting
        mpp_lite                    cell center is the only interior point
        aposteriori_limiting        whether to call trouble detection and 2d fallback
        convex                      a more mpp version of a posteriori limiting
        cause_trouble               set all cells to be troubled, forcing 2d fallback
        SED                         whether to enable smooth extrema detection
        NAD                         simulation NAD tolerance
                                        set to None or +inf to disable NAD
        PAD                         physical admissibility detection (lower, upper)
                                        set to None or (-inf, +inf) to disable PAD
        visualization_tolerance     tolerance for whether to visualize theta and/or
                                        troubled cells based on some violation
                                        set to None or -inf to visualize simulation
                                        values
        adjust_time_step            whether to reduce timestep for order >4
        modify_time_step            whether to conditionally reduce dt by half
        log_every                   number of iterations to complete before logging
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
        const: float = 0.0,
        n: tuple = (32, 32),
        x: tuple = (0, 1),
        y: tuple = None,
        t0: float = 0,
        snapshot_dt: float = 1.0,
        num_snapshots: int = 2,
        v: tuple = (1, 1),
        courant: float = 0.5,
        order: int = 1,
        flux_strategy: str = "gauss-legendre",
        apriori_limiting: bool = False,
        mpp_lite: bool = False,
        aposteriori_limiting: bool = False,
        convex: bool = False,
        cause_trouble: bool = False,
        SED: bool = False,
        NAD: float = 0.0,
        PAD: tuple = (0, 1),
        visualization_tolerance: float = None,
        adjust_time_step: bool = False,
        modify_time_step: bool = False,
        log_every: int = 100000,
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
            convex,
            cause_trouble,
            SED,
            NAD,
            PAD,
            visualization_tolerance,
            adjust_time_step,
            modify_time_step,
            log_every,
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

        # boundary conditon
        if bc == "periodic":
            self.apply_bc = self.apply_periodic_bc
        if bc == "dirichlet":
            self.apply_bc = self.apply_dirichlet_bc
        self.const = const

        # initialize limiting
        self.apriori_limiting = apriori_limiting
        self.mpp_lite = mpp_lite
        self.aposteriori_limiting = aposteriori_limiting
        self.convex = convex

        # arrays for storing local min/max
        self.m = 0 * self.ones_f
        self.M = 0 * self.ones_f

        # initialize cause_trouble and udot_evaluation_count
        self.cause_trouble = 1 if cause_trouble else 0
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

        # stencils: right/left conservative interpolation from a volume or line segment
        left_interface_stencil = ConservativeInterpolation.construct_from_order(
            order, "left"
        ).nparray()
        right_interface_stencil = ConservativeInterpolation.construct_from_order(
            order, "right"
        ).nparray()
        self._conservative_stencil_size = len(left_interface_stencil)
        self._conservative_k = int(
            np.floor(len(left_interface_stencil) / 2)
        )  # cell reach of stencil

        if self.mpp_lite:
            cell_center_stencil = ConservativeInterpolation.construct_from_order(
                order, "center"
            ).nparray()
            while cell_center_stencil.size < self._conservative_stencil_size:
                cell_center_stencil = np.concatenate(
                    (np.zeros(1), cell_center_stencil, np.zeros(1))
                )
            self.cell_center_stencil = cell_center_stencil[np.newaxis]

        # quadrature points setup
        self.flux_strategy = flux_strategy if self.ndim > 1 else "gauss-legendre"
        if self.flux_strategy == "gauss-legendre":
            p = order - 1  # degree of reconstructed polynomial
            N_G = int(
                np.ceil((p + 1) / 2)
            )  # number of gauss-legendre quadrature points
            N_GL = int(
                np.ceil((p + 3) / 2)
            )  # number of gauss-lobatto quadrature points
            # guass-legendre quadrature
            (
                gauss_quadr_points,
                gauss_quadr_weights,
            ) = np.polynomial.legendre.leggauss(N_G)
            # transform to cell coordinate
            gauss_quadr_points /= 2
            gauss_quadr_weights /= 2
            self.gauss_quadr_weights = np.array(
                gauss_quadr_weights
            )  # for evaluating line integrals
            # reconstruct polynomial and evaluate at each quadrature point
            list_of_line_stencils = []
            for x in gauss_quadr_points:
                stencil = ConservativeInterpolation.construct_from_order(
                    order, x
                ).nparray()
                # if the stencil is short, assume it needs a 0 on either end
                while len(stencil) < self._conservative_stencil_size:
                    stencil = np.concatenate((np.zeros(1), stencil, np.zeros(1)))
                assert len(stencil) == self._conservative_stencil_size
                list_of_line_stencils.append(stencil)  # ordered from left to right
            self.line_stencils = np.array(list_of_line_stencils)

            # stencils for reconstructing pointwise values along a line average
            list_of_interior_pointwise_stencils = []
            if N_GL > 2:
                # interpolating values along line segments
                GL_quadr_points, _ = gauss_lobatto(N_GL)
                # scale to cell of width 1
                interior_GL_quadr_points = GL_quadr_points[1:-1] / 2
                # cell center is the only central point considered for mpp_lite
                if self.mpp_lite:
                    interior_GL_quadr_points = []
                # generate stencils two are given
                for x in interior_GL_quadr_points:
                    stencil = ConservativeInterpolation.construct_from_order(
                        order, x
                    ).nparray()
                    # if the stencil is short, assume it needs a 0 on either end
                    while len(stencil) < self._conservative_stencil_size:
                        stencil = np.concatenate((np.zeros(1), stencil, np.zeros(1)))
                    assert len(stencil) == self._conservative_stencil_size
                    list_of_interior_pointwise_stencils.append(stencil)
            # assort list of stencils for pointwise interpolation
            self.pointwise_stencils = np.array(
                [left_interface_stencil]
                + list_of_interior_pointwise_stencils
                + [right_interface_stencil]
            )
            if self.apriori_limiting:
                _, GL_quadr_weights = gauss_lobatto(N_GL)
                C_mpp = min(GL_quadr_weights) / 2
                self.dt_min = C_mpp / v_over_h
                # check if timestep is small enough for mpp
                if self.dt * v_over_h > C_mpp:
                    print(
                        "WARNING: Maximum principle preserving not satisfied.\nTry a ",
                        f"courant condition less than {C_mpp}\n",
                    )
            # no transverse integral stencil
            self._transverse_k = 0
        elif self.flux_strategy == "transverse":
            gauss_quadr_points = np.array([0.0])
            self.line_stencils = ConservativeInterpolation.construct_from_order(
                order, "center"
            ).nparray()[np.newaxis]
            self.pointwise_stencils = np.array(
                [left_interface_stencil] + [right_interface_stencil]
            )

            # transverse integral stencil
            self.transverse_stencil = TransverseIntegral.construct_from_order(
                order
            ).nparray()
            self._transverse_stencil_size = len(self.transverse_stencil)
            self._transverse_k = int(
                np.floor(len(self.transverse_stencil) / 2)
            )  # cell reach of stencil

        # ghost width
        self.riemann_zone = max(self._transverse_k, 1)
        self.gw_riemann = self.riemann_zone + self._conservative_k
        if self.flux_strategy == "gauss-legendre":
            self.gw_nonriemann = self.riemann_zone + self._conservative_k
        elif self.flux_strategy == "transverse":
            self.gw_nonriemann = self.riemann_zone + self._transverse_k
        # the pointwise data will have a uniform ghost width, so we must keep track of
        # the excess in the riemann and non-riemann directions
        self.excess_riemann_gw = max(self.riemann_zone - 1, 0)
        self.excess_nonriemann_gw = max(self.riemann_zone - self._transverse_k, 0)

        # velocity at cell interfaces
        if self.ndim == 1:
            self.a = v * np.ones(nx + 1)
        if self.ndim == 2:
            # x and y values at interface points
            na = np.newaxis
            # EW interface
            EW_interface_x, EW_interface_y = np.meshgrid(self.x_interface, self.y)
            EW_interface_x = np.repeat(
                EW_interface_x[na], len(gauss_quadr_points), axis=0
            )
            EW_interface_y = np.repeat(
                EW_interface_y[na], len(gauss_quadr_points), axis=0
            )
            EW_interface_y += gauss_quadr_points[:, na, na] * self.hy
            EW_midpoint_x, EW_midpoint_y = np.meshgrid(self.x_interface, self.y)
            # NS interface
            NS_interface_x, NS_interface_y = np.meshgrid(self.x, self.y_interface)
            NS_interface_x = np.repeat(
                NS_interface_x[na], len(gauss_quadr_points), axis=0
            )
            NS_interface_y = np.repeat(
                NS_interface_y[na], len(gauss_quadr_points), axis=0
            )
            NS_interface_x += gauss_quadr_points[:, na, na] * self.hx
            NS_midpoint_x, NS_midpoint_y = np.meshgrid(self.x, self.y_interface)
            # evaluate v components normal to cell interfaces
            if isinstance(v, tuple):
                self.a = v[0] * np.ones(EW_interface_y.shape)
                self.a_midpoint = v[0] * np.ones(EW_midpoint_x.shape)
                if len(v) < 2:
                    self.b = v[0] * np.ones(NS_interface_x.shape)
                    self.b_midpoint = v[0] * np.ones(NS_midpoint_y.shape)
                else:
                    self.b = v[1] * np.ones(NS_interface_x.shape)
                    self.b_midpoint = v[1] * np.ones(NS_midpoint_y.shape)
            if callable(v):
                self.a = v(EW_interface_x, EW_interface_y)[0]
                self.b = v(NS_interface_x, NS_interface_y)[1]
                self.a_midpoint = v(EW_midpoint_x, EW_midpoint_y)[0]
                self.b_midpoint = v(NS_midpoint_x, NS_midpoint_y)[1]
            if self.flux_strategy == "transverse":  # apply boundary for transverse flux
                self.a = self.apply_bc(
                    self.a, gw=[(0,), (self._transverse_k,), (0,)], const=self.const
                )
                self.b = self.apply_bc(
                    self.b, gw=[(0,), (0,), (self._transverse_k,)], const=self.const
                )

        # fluxes
        if self.ndim == 1:
            self.f = np.zeros(nx + 1)
            self.g = np.zeros(nx + 1)
        if self.ndim == 2:
            self.f = np.zeros((ny, nx + 1))  # east/west fluxes
            self.g = np.zeros((ny + 1, nx))  # north/south fluxes

        # dynamic function assignment
        if self.ndim == 1:
            self.get_fluxes = self.get_fluxes_1d
            self.f_of_neighbors = f_of_3_neighbors
            self.compute_alpha = self.compute_alpha_1d
        if self.ndim == 2:
            self.get_fluxes = self.get_fluxes_2d
            self.f_of_neighbors = f_of_4_neighbors
            self.compute_alpha = self.compute_alpha_2d

        if modify_time_step:
            self.looks_good = self.check_mpp

        if self.flux_strategy == "gauss-legendre":
            self.integrate_fluxes = self.gauss_legendre_integral
        elif self.flux_strategy == "transverse":
            self.integrate_fluxes = self.transverse_integral

        if self.apriori_limiting:
            self.apriori_limiter = self.mpp_limiter
            if self.mpp_lite:
                if self.ndim == 1:
                    self.mpplite_Mm_adjustment = self.mpplite_Mm_adjustment_1d
                if self.ndim == 2:
                    self.mpplite_Mm_adjustment = self.mpplite_Mm_adjustment_2d
            else:
                self.mpplite_Mm_adjustment = self.no_mpplite_Mm_adjustment
        else:
            self.apriori_limiter = self.trivial_limiter

        if SED:
            self.detect_smooth_extrema = detect_smooth_extrema
        else:
            self.detect_smooth_extrema = detect_no_smooth_extrema

        if self.aposteriori_limiting:
            self.aposteriori_limiter = self.fallback_scheme
            if self.convex:
                if self.ndim == 1:
                    self.revise_fluxes = self.revise_blended_fluxes_1d
                elif self.ndim == 2:
                    self.revise_fluxes = self.revise_blended_fluxes_2d
            else:
                if self.ndim == 1:
                    self.revise_fluxes = self.revise_fluxes_1d
                elif self.ndim == 2:
                    self.revise_fluxes = self.revise_fluxes_2d
        else:
            self.aposteriori_limiter = self.trivial_limiter

    def get_dynamics(self) -> np.ndarray:
        """
        advection equation:
        dudt + dfdx + dgdy = 0
        f and g are fluxes in x and y
        """
        return -self.hx_recip * (
            self.f[..., 1:] - self.f[..., :-1]
        ) + -self.hy_recip * (self.g[1:, ...] - self.g[:-1, ...])

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
        self.get_fluxes(u=u)  # high order flux update
        self.aposteriori_limiter(u=u, dt=dt)
        return self.get_dynamics()

    def apply_periodic_bc(
        self, u_without_ghost_cells: np.ndarray, gw: {tuple, list, int}, **kwargs
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
        returns:
            u with periodic ghost zones
        """
        return np.pad(u_without_ghost_cells, pad_width=gw, mode="wrap")

    def apply_dirichlet_bc(
        self, u_without_ghost_cells: np.ndarray, gw: {tuple, list, int}, const, **kwargs
    ) -> np.ndarray:
        return np.pad(
            u_without_ghost_cells, pad_width=gw, mode="constant", constant_values=const
        )

    def conservative_interpolation(
        self, u: np.ndarray, stencils: np.ndarray, axis: int = 0
    ):
        """
        args:
            u           array of arbitrary shape
            stensils    m, n array
                            m   number of stencils
                            n   stencil size
            axis
        returns:
            u with new first axis of size m (the number of interpolated values)
        """
        stacked_u = stack(u, stacks=stencils.shape[1], axis=axis)
        interpolations = np.apply_along_axis(
            lambda stencil: apply_stencil(stacked_u, stencil), axis=1, arr=stencils
        )
        return interpolations

    def riemann(
        self,
        v: np.ndarray,
        left_value: np.ndarray,
        right_value: np.ndarray,
    ) -> float:
        """
        args:
            arrays of arbitrary shape
            v           advection velocity defined at an interface
            left_value  value to the left of the interface
            right_value value to the right of the interface
        returns:
            array of same shape
            pointwise fluxes chosen based on the sign of v
        """
        left_flux, right_flux = v * left_value, v * right_value
        return ((right_flux + left_flux) - np.abs(v) * (right_value - left_value)) / 2.0

    def get_fluxes_1d(self, u: np.ndarray):
        """
        args:
            u   (nx,)
        overwrites:
            self.a  (nx + 1,)
        """
        # interpolate points from line averages
        u_gw = self.apply_bc(u, gw=self.gw_riemann, const=self.const)
        points = self.conservative_interpolation(
            u=u_gw,
            stencils=self.pointwise_stencils,
        )
        # evaluate alpha and theta
        fallback = self.apply_bc(u, gw=4, const=self.const)
        alpha = self.compute_alpha(fallback)
        fallback = fallback[2:-2]  # remove extra gw for for computing smooth extrema
        m = self.f_of_neighbors(fallback, f=np.minimum)
        M = self.f_of_neighbors(fallback, f=np.maximum)
        self.m[...], self.M[...] = m[1:-1], M[1:-1]
        fallback = fallback[1:-1]  # remove extra gw for computing m and M
        theta, meaningful_theta, M_i, m_i = self.apriori_limiter(
            points=points, u=fallback, m=m, M=M
        )
        # apply smooth extrema detection
        theta = np.where(
            np.logical_or(m_i < self.PAD[0], M_i > self.PAD[1]),
            theta,
            np.where(alpha < 1, theta, 1.0),
        )
        # store theta visualization
        self.theta += theta[1:-1]
        self.visualized_theta += np.where(meaningful_theta, theta, 1)[1:-1]
        # limit slopes
        points = theta * (points - fallback) + fallback
        # riemann problem
        right_points = points[-1, :-1]
        left_points = points[0, 1:]
        self.f[...] = self.riemann(
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
        # gauss-legendre interpolation to get line averages from cell averages
        horizontal_lines = self.conservative_interpolation(
            u=self.apply_bc(
                u, gw=((self.gw_nonriemann,), (self.gw_riemann,)), const=self.const
            ),
            stencils=self.line_stencils,
            axis=0,
        )
        vertical_lines = self.conservative_interpolation(
            u=self.apply_bc(
                u, gw=((self.gw_riemann,), (self.gw_nonriemann,)), const=self.const
            ),
            stencils=self.line_stencils,
            axis=1,
        )
        # gauss-legendre interpolation to get pointise values from line averages
        horizontal_points = self.conservative_interpolation(
            u=horizontal_lines, stencils=self.pointwise_stencils, axis=2
        )
        vertical_points = self.conservative_interpolation(
            u=vertical_lines, stencils=self.pointwise_stencils, axis=1
        )
        # evaluate alpha and theta
        fallback = self.apply_bc(u, gw=self.riemann_zone + 3, const=self.const)
        alpha = self.compute_alpha(fallback)
        fallback = fallback[2:-2, 2:-2]  # remove extra gw for for computing SE
        m = self.f_of_neighbors(fallback, f=np.minimum)
        M = self.f_of_neighbors(fallback, f=np.maximum)
        self.m[...] = chop(
            m, chop_size=self.riemann_zone, axis=[0, 1]
        )  # store for later
        self.M[...] = chop(
            M, chop_size=self.riemann_zone, axis=[0, 1]
        )  # store for later
        fallback = fallback[1:-1, 1:-1]  # remove extra gw for computing m and M
        theta, meaningful_theta, M_ij, m_ij = self.apriori_limiter(
            points=np.array([horizontal_points, vertical_points]), u=fallback, m=m, M=M
        )
        # apply smooth extrema detection
        theta = np.where(
            np.logical_or(m_ij < self.PAD[0], M_ij > self.PAD[1]),
            theta,
            np.where(alpha < 1, theta, 1.0),
        )
        # store theta
        stored_theta = chop(theta, self.riemann_zone, axis=0)
        stored_theta = chop(stored_theta, self.riemann_zone, axis=1)
        self.theta += stored_theta
        visualized_theta = np.where(meaningful_theta, theta, 1)
        visualized_theta = chop(visualized_theta, self.riemann_zone, axis=0)
        visualized_theta = chop(visualized_theta, self.riemann_zone, axis=1)
        self.visualized_theta += visualized_theta
        # limit slopes
        horizontal_points = theta * (horizontal_points - fallback) + fallback
        vertical_points = theta * (vertical_points - fallback) + fallback
        # remove excess due to uniform ghost zone
        horizontal_points = chop(horizontal_points, self.excess_riemann_gw, axis=3)
        horizontal_points = chop(horizontal_points, self.excess_nonriemann_gw, axis=2)
        vertical_points = chop(vertical_points, self.excess_riemann_gw, axis=2)
        vertical_points = chop(vertical_points, self.excess_nonriemann_gw, axis=3)
        # riemann problem
        ns_pointwise_fluxes = self.riemann(
            v=self.b,
            left_value=vertical_points[-1, :, :-1, :],  # north points
            right_value=vertical_points[0, :, 1:, :],  # south points
        )
        ew_pointwise_fluxes = self.riemann(
            v=self.a,
            left_value=horizontal_points[-1, :, :, :-1],  # east points
            right_value=horizontal_points[0, :, :, 1:],  # west points
        )
        # integrate fluxes with gauss-legendre quadrature
        self.f[...] = self.integrate_fluxes(ew_pointwise_fluxes, axis=1)
        self.g[...] = self.integrate_fluxes(ns_pointwise_fluxes, axis=2)

    def compute_alpha_1d(self, u):
        """
        args:
            u   np array (m + 6,)
        returns
            alpha   np array (m,)
        """
        return self.detect_smooth_extrema(u, h=self.hx, axis=0)

    def compute_alpha_2d(self, u):
        """
        args:
            u   np array (m + 6, n + 6)
        returns
            alpha   np array (m, n)
        """
        alpha_x = self.detect_smooth_extrema(u, h=self.hx, axis=1)[3:-3, :]
        alpha_y = self.detect_smooth_extrema(u, h=self.hy, axis=0)[:, 3:-3]
        alpha = np.where(alpha_x < alpha_y, alpha_x, alpha_y)
        return alpha

    def no_mpplite_Mm_adjustment(self, u, M_ij, m_ij):
        """
        args:
            u               np array (m, n)
            M_ij            np array (m,) or (m, n),    maximum of reconstructed points
                                                        not including cell center
            m_ij            np array (m,) or (m, n),    minimum of reconstructed points
                                                        not including cell center
        """
        return M_ij, m_ij

    def mpplite_Mm_adjustment_1d(self, u, M_ij, m_ij):
        u_gw = self.apply_bc(u[1:-1], gw=self.gw_riemann, const=self.const)
        cell_center_points = self.conservative_interpolation(
            u=u_gw,
            stencils=self.cell_center_stencil,
        )[
            0
        ]  # remove ghost width to retrive original array u
        M_ij = np.maximum(M_ij, cell_center_points)
        m_ij = np.minimum(m_ij, cell_center_points)
        return M_ij, m_ij

    def mpplite_Mm_adjustment_2d(self, u, M_ij, m_ij):
        horizontal_lines = self.conservative_interpolation(
            u=self.apply_bc(
                chop(u, self.excess_riemann_gw + 1, axis=[0, 1]),
                gw=((self.gw_riemann,), (self.gw_riemann,)),
                const=self.const,
            ),
            stencils=self.cell_center_stencil,
            axis=0,
        )
        cell_center_points = self.conservative_interpolation(
            u=horizontal_lines, stencils=self.cell_center_stencil, axis=2
        )[
            0, 0
        ]  # remove ghost width to retrive original array u
        M_ij = np.maximum(M_ij, cell_center_points)
        m_ij = np.minimum(m_ij, cell_center_points)
        return M_ij, m_ij

    def gauss_legendre_integral(self, pointwise_fluxes: np.ndarray, axis: int):
        """
        args:
            pointwise_fluxes    pointwise values on a gauss-legendre quadrature along a
                                cell face
        returns:
            face integral approximated with gauss-legendre quadrature weights
        """
        return apply_stencil(pointwise_fluxes, self.gauss_quadr_weights)

    def transverse_integral(self, pointwise_fluxes: np.ndarray, axis: int):
        """
        args:
            pointwise_fluxes    cell midpoint values
        returns:
            face integral approximated with a transverse lagrage integral
        """
        pointwise_fluxes_stack = stack(
            pointwise_fluxes, stacks=self._transverse_stencil_size, axis=axis
        )
        return apply_stencil(pointwise_fluxes_stack, self.transverse_stencil)

    def trivial_limiter(self, u, **kwargs):
        return np.ones_like(u), np.ones_like(u), np.ones_like(u), np.ones_like(u)

    def mpp_limiter(
        self, points: np.ndarray, u: np.ndarray, m: np.ndarray, M: np.ndarray
    ):
        """
        args:
            points  pointwise values (..., m) or (..., m, n)
            u       fallback values (m,) or (m, n)
            m       min of cell and neighbors (m,) or (m, n)
            M       max of cell and neighbors (m,) or (m, n)
        returns:
            theta           np array (m,) or (m, n), reconstruction limiter
            meaningful_theta np array (m,) or (m, n), boolean intolerated violations
            M_ij            np array (m,) or (m, n), maximum of reconstructed points
            m_ij            np array (m,) or (m, n), minimum of reconstructed points
        """
        # max and min of u evaluated at quadrature points
        M_ij = np.amax(points, axis=tuple(range(points.ndim - self.ndim)))
        m_ij = np.amin(points, axis=tuple(range(points.ndim - self.ndim)))
        M_ij, m_ij = self.mpplite_Mm_adjustment(u, M_ij, m_ij)

        # theta visualization
        u_range = np.max(u) - np.min(u)
        meaningful_theta = np.where(
            m_ij - m < -self.visualization_tolerance * u_range, 1, 0
        )
        meaningful_theta[...] = np.where(
            M_ij - M > self.visualization_tolerance * u_range, 1, meaningful_theta
        )
        # evaluate slope limiter
        theta = np.ones_like(u)
        M_arg = np.abs(M - u) / np.maximum(np.abs(M_ij - u), 1e-16 * np.ones_like(u))
        m_arg = np.abs(m - u) / np.maximum(np.abs(m_ij - u), 1e-16 * np.ones_like(u))
        theta = np.where(M_arg < theta, M_arg, theta)
        theta = np.where(m_arg < theta, m_arg, theta)

        return theta, meaningful_theta, M_ij, m_ij

    def fallback_scheme(self, u: np.ndarray, dt: float):
        """
        args:
            u   initial state   (m, n)
            ucandidate  proposed next state (ny, nx)
        overwrites:
            self.NS_fluxes and self.EW_fluxes to fall back to second order
            when trouble is detected
        """
        # reset trouble array

        trouble = np.zeros_like(self.u0, dtype="int")

        # compute candidate solution
        unew = u + dt * self.get_dynamics()

        # NAD
        u_range = np.max(u) - np.min(u)
        tolerance = self.NAD * u_range
        upper_differences, lower_differences = unew - self.M, unew - self.m
        possible_trouble = np.where(lower_differences < -tolerance, 1, 0)
        possible_trouble = np.where(upper_differences > tolerance, 1, possible_trouble)

        # SED
        alpha = self.compute_alpha(self.apply_bc(unew, gw=3, const=self.const))

        # PAD then SED then NAD
        trouble = np.where(
            np.logical_or(unew < self.PAD[0], unew > self.PAD[1]),
            1,
            np.where(alpha < 1, possible_trouble, 0),
        )

        # set all cells to 1 if cause_trouble = True
        trouble = (
            1 - self.cause_trouble
        ) * trouble + self.cause_trouble * np.ones_like(self.u0, dtype="int")

        # store history of troubled cells
        self.trouble += trouble
        visualized_trouble = np.where(
            lower_differences < -self.visualization_tolerance * u_range, 1, 0
        )
        visualized_trouble = np.where(
            upper_differences > self.visualization_tolerance * u_range,
            1,
            visualized_trouble,
        )
        self.visualized_trouble[...] = np.where(
            visualized_trouble, 1, self.visualized_trouble
        )

        # revise fluxes of troubled cells
        self.revise_fluxes(u, trouble)

    def revise_fluxes_1d(self, u, trouble):
        """
        args:
            u   (m,)
            trouble (m,)
        overwrites:
            self.f  (m + 1,)
        """

        # flag faces of troubled cells as troubled
        troubled_faces = np.zeros_like(self.f, dtype=int)
        troubled_faces[:-1] = trouble
        troubled_faces[1:] = np.where(trouble, 1, troubled_faces[1:])

        # compute second order face interpolations
        u_2gw = self.apply_bc(u, gw=2, const=self.const)
        left_face, right_face = compute_fallback_faces(u_2gw, axis=0)

        # revise fluxes
        fallback_fluxes = self.riemann(
            v=self.a, left_value=right_face[:-1], right_value=left_face[1:]
        )
        self.f[...] = np.where(troubled_faces, fallback_fluxes, self.f)

    def revise_fluxes_2d(self, u, trouble):
        """
        args:
            u   (m, n)
            trouble (m, n)
        overwrites:
            self.f  (m, n + 1)
            self.g  (m + 1, n)
        """

        # flag faces of troubled cells as troubled
        NS_troubled_faces = np.zeros_like(self.g, dtype="int")
        EW_troubled_faces = np.zeros_like(self.f, dtype="int")
        NS_troubled_faces[:-1, :] = trouble
        NS_troubled_faces[1:, :] = np.where(trouble, 1, NS_troubled_faces[1:, :])
        EW_troubled_faces[:, :-1] = trouble
        EW_troubled_faces[:, 1:] = np.where(trouble, 1, EW_troubled_faces[:, 1:])

        # compute second order face interpolations
        u_2gw = self.apply_bc(u, gw=2, const=self.const)
        south_face, north_face = compute_fallback_faces(u_2gw[:, 2:-2], axis=0)
        west_face, east_face = compute_fallback_faces(u_2gw[2:-2, :], axis=1)

        # revise fluxes
        NS_fallback_fluxes = self.riemann(
            v=self.b_midpoint,
            left_value=north_face[:-1, :],
            right_value=south_face[1:, :],
        )
        EW_fallback_fluxes = self.riemann(
            v=self.a_midpoint,
            left_value=east_face[:, :-1],
            right_value=west_face[:, 1:],
        )
        self.f[...] = np.where(EW_troubled_faces, EW_fallback_fluxes, self.f)
        self.g[...] = np.where(NS_troubled_faces, NS_fallback_fluxes, self.g)

    def revise_blended_fluxes_1d(self, u: np.ndarray, trouble: np.ndarray):
        """
        args:
            u   (m,)
            trouble (m,)
        overwrites:
            self.f  (m + 1,)
        """
        # initialize theta
        trouble_2gw = self.apply_bc(trouble, gw=2, const=self.const)
        theta = trouble_2gw.astype("float")

        # First neighbors
        theta[:-1] = np.maximum(0.75 * trouble_2gw[1:], theta[:-1])
        theta[1:] = np.maximum(0.75 * trouble_2gw[:-1], theta[1:])

        # Second neighbors
        theta[:-1] = np.maximum(0.25 * (theta[1:] > 0), theta[:-1])
        theta[1:] = np.maximum(0.25 * (theta[:-1] > 0), theta[1:])

        # flag affected faces with theta
        affected_face_x = np.maximum(theta[1:-2], theta[2:-1])

        # compute second order face interpolations
        u_2gw = self.apply_bc(u, gw=2, const=self.const)
        left_face, right_face = compute_fallback_faces(u_2gw, axis=0)

        # revise fluxes
        fallback_fluxes_x = self.riemann(
            v=self.a,
            left_value=right_face[:-1],
            right_value=left_face[1:],
        )

        self.f[...] = (
            affected_face_x * fallback_fluxes_x + (1 - affected_face_x) * self.f
        )

    def revise_blended_fluxes_2d(self, u: np.ndarray, trouble: np.ndarray):
        """
        args:
            u   (m, n)
            trouble (m, n)
        overwrites:
            self.f  (m, n + 1)
            self.g  (m + 1, n)
        """
        # initialize theta
        trouble_2gw = self.apply_bc(trouble, gw=2, const=self.const)
        theta = trouble_2gw.astype("float")

        # First neighbors
        theta[:, :-1] = np.maximum(0.75 * trouble_2gw[:, 1:], theta[:, :-1])
        theta[:, 1:] = np.maximum(0.75 * trouble_2gw[:, :-1], theta[:, 1:])
        theta[:-1, :] = np.maximum(0.75 * trouble_2gw[1:, :], theta[:-1, :])
        theta[1:, :] = np.maximum(0.75 * trouble_2gw[:-1, :], theta[1:, :])
        # Second neighbors
        theta[:-1, :-1] = np.maximum(0.5 * trouble_2gw[1:, 1:], theta[:-1, :-1])
        theta[:-1, 1:] = np.maximum(0.5 * trouble_2gw[1:, :-1], theta[:-1, 1:])
        theta[1:, :-1] = np.maximum(0.5 * trouble_2gw[:-1, 1:], theta[1:, :-1])
        theta[1:, 1:] = np.maximum(0.5 * trouble_2gw[:-1, :-1], theta[1:, 1:])
        # Third neighbors
        theta[:, :-1] = np.maximum(0.25 * (theta[:, 1:] > 0), theta[:, :-1])
        theta[:, 1:] = np.maximum(0.25 * (theta[:, :-1] > 0), theta[:, 1:])
        theta[:-1, :] = np.maximum(0.25 * (theta[1:, :] > 0), theta[:-1, :])
        theta[1:, :] = np.maximum(0.25 * (theta[:-1, :] > 0), theta[1:, :])

        # flag affected faces with theta
        affected_face_x = np.maximum(theta[2:-2, 1:-2], theta[2:-2, 2:-1])
        affected_face_y = np.maximum(theta[1:-2, 2:-2], theta[2:-1, 2:-2])

        # compute second order face interpolations
        u_2gw = self.apply_bc(u, gw=2, const=self.const)
        south_face, north_face = compute_fallback_faces(u_2gw[:, 2:-2], axis=0)
        west_face, east_face = compute_fallback_faces(u_2gw[2:-2, :], axis=1)

        # revise fluxes
        fallback_fluxes_x = self.riemann(
            v=self.a_midpoint,
            left_value=east_face[:, :-1],
            right_value=west_face[:, 1:],
        )
        fallback_fluxes_y = self.riemann(
            v=self.b_midpoint,
            left_value=north_face[:-1, :],
            right_value=south_face[1:, :],
        )

        self.f[...] = (
            affected_face_x * fallback_fluxes_x + (1 - affected_face_x) * self.f
        )
        self.g[...] = (
            affected_face_y * fallback_fluxes_y + (1 - affected_face_y) * self.g
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
        tol = 1e-10
        return not np.logical_or(
            np.any(u < self.PAD[0] - tol), np.any(u > self.PAD[1] + tol)
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
