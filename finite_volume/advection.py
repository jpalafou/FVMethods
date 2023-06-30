# advection solution schems
import numpy as np
import os
import pickle
import warnings
from finite_volume.initial_conditions import generate_ic
from finite_volume.integrate import Integrator
from finite_volume.fvscheme import ConservativeInterpolation, TransverseIntegral
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


warnings.filterwarnings("ignore")


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
        const                   for Neumann bc
        flux_strategy           'gauss-legendre' or 'transverse'
        apriori_limiting        whether to follow zhang and shu mpp limiting
        aposteriori_limiting    whether to call trouble detection and 2d fallback
        smooth_extrema_detection    whether to hold back on limiting extrema if smooth
        cause_trouble           set all cells to be troubled, forcing 2d fallback
        NAD                     simulation NAD tolerance
                                set to None or +inf to disable NAD
        PAD                     physical admissibility detection bounds (lower, upper)
                                set to None or (-inf, +inf) to disable PAD
        visualization_tolerance tolerance for whether to visualize theta or a troubled
                                cell based on some violation
                                set to None or -inf to visualize simulation values
        loglen                  number of saved states
        adjust_time_step        whether to reduce timestep for order >4
        load                    whether to load precalculated solution or do it again
    returns:
        u   array of saved states
    """

    def __init__(
        self,
        u0: str = "square",
        bc: str = "periodic",
        n: tuple = (32, 32),
        x: tuple = (0, 1),
        y: tuple = None,
        t0: float = 0,
        T: float = 1,
        v: tuple = (1, 1),
        courant: float = 0.5,
        order: int = 1,
        const: float = 0.0,
        flux_strategy: str = "gauss-legendre",
        apriori_limiting: bool = False,
        aposteriori_limiting: bool = False,
        smooth_extrema_detection: bool = False,
        cause_trouble: bool = False,
        NAD: float = 0.0,
        PAD: tuple = None,
        visualization_tolerance: float = None,
        log_every: int = 10,
        adjust_time_step: bool = False,
        modify_time_step: bool = False,
        load: bool = True,
        load_directory: str = "data/solutions/",
    ):
        # create filename out of the initialization arguments
        self.load = load
        u0_str = u0.__name__ if callable(u0) else str(u0)
        v_str = v.__name__ if callable(v) else str(v)
        filename_components = [
            u0_str,
            bc,
            n,
            x,
            y,
            t0,
            T,
            v_str,
            courant,
            order,
            const,
            flux_strategy,
            apriori_limiting,
            aposteriori_limiting,
            smooth_extrema_detection,
            cause_trouble,
            NAD,
            PAD,
            visualization_tolerance,
            log_every,
            adjust_time_step,
            modify_time_step,
        ]
        self._filename = "_".join(str(component) for component in filename_components)
        self._load_directory = load_directory

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
            self.hy = 1
            self.Ly = 1
            self.hy_recip = 1
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
        self.courant = courant
        self.adjust_time_step = adjust_time_step
        self.order = order
        v_over_h = vx_max / self.hx + vy_max / self.hy
        if v_over_h == 0:
            print("0 velocity case: setting v / h to 10 / T")
            v_over_h = 10 / T
        dt = courant / v_over_h
        dt_adjustment = None
        if adjust_time_step and order > 4:
            dt_adjustment_x = rk4_dt_adjust(self.hx, self.Lx, order)
            dt_adjustment_y = rk4_dt_adjust(self.hy, self.Ly, order)
            dt_adjustment = min(dt_adjustment_x, dt_adjustment_y)
            dt = dt * dt_adjustment
            print(
                f"Decreasing timestep by a factor of {dt_adjustment} to maintain",
                f" order {order} with rk4",
            )
        self.dt_adjustment = dt_adjustment

        if modify_time_step:
            self.looks_good = self.check_mpp

        # initial condition
        if isinstance(u0, str):
            u0 = generate_ic(type=u0, x=self.x, y=self.y)
        if callable(u0):
            if self.ndim == 1:
                u0 = u0(x=self.x)
            if self.ndim == 2:
                u0 = u0(x=self.x, y=self.y)
        self.u0_min = np.min(u0)
        self.u0_max = np.max(u0)

        # initialize integrator
        super().__init__(u0=u0, T=T, dt=dt, t0=t0, log_every=log_every)

        # boundary conditon
        if bc == "periodic":
            self.apply_bc = self.apply_periodic_bc
        if bc == "neumann":
            self.apply_bc = self.apply_neumann_bc
        self.const = const

        # initialize limiting
        self.apriori_limiting = apriori_limiting
        self.aposteriori_limiting = aposteriori_limiting

        # arrays for storing local min/max
        self.m = np.zeros(self.u0.shape)
        self.M = np.zeros(self.u0.shape)

        # arrays for storing theta
        self.theta = np.zeros_like(self.u0, dtype="float64")
        self.theta_history = np.ones_like(self.u, dtype="float64")
        self.visualize_theta = np.ones_like(self.u0, dtype="int")
        self.visualize_theta_history = np.ones_like(self.u, dtype="int")

        # arrays for storing troubled cells
        self.trouble = np.zeros_like(self.u0, dtype="int")
        self.trouble_history = np.zeros_like(self.u, dtype="float64")
        self.visualize_trouble = np.ones_like(self.u0, dtype="int")
        self.visualize_trouble_history = np.ones_like(self.u, dtype="int")

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
                (
                    interior_GL_quadr_points,
                    _,
                ) = np.polynomial.legendre.leggauss(N_GL - 2)
                # scale to cell of width 1
                interior_GL_quadr_points /= 2
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
                # endpoint GL quadrature weights
                endpoint_GL_quadr_weights = 2 / (N_GL * (N_GL - 1))
                endpoint_GL_quadr_weights /= 2  # scale to cell of width 1
                # check if timestep is small enough for mpp
                if self.dt * v_over_h > endpoint_GL_quadr_weights:
                    print(
                        "WARNING: Maximum principle preserving not satisfied.\nTry a ",
                        f"courant condition less than {endpoint_GL_quadr_weights}\n",
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
            self.revise_fluxes = self.revise_fluxes_1d
        if self.ndim == 2:
            self.get_fluxes = self.get_fluxes_2d
            self.f_of_neighbors = f_of_4_neighbors
            self.compute_alpha = self.compute_alpha_2d
            self.revise_fluxes = self.revise_fluxes_2d

        if self.flux_strategy == "gauss-legendre":
            self.integrate_fluxes = self.gauss_legendre_integral
        elif self.flux_strategy == "transverse":
            self.integrate_fluxes = self.transverse_integral

        if self.apriori_limiting:
            self.apriori_limiter = self.mpp_limiter
        else:
            self.apriori_limiter = self.trivial_limiter

        if smooth_extrema_detection:
            self.detect_smooth_extrema = detect_smooth_extrema
        else:
            self.detect_smooth_extrema = detect_no_smooth_extrema

        if self.aposteriori_limiting:
            self.aposteriori_limiter = self.fallback_scheme
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

    def apply_neumann_bc(
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
        theta, visualize_theta = self.apriori_limiter(
            points=points, u=fallback, m=m, M=M
        )
        # apply smooth extrema detection
        theta = np.where(alpha < 1, theta, 1.0)
        # store theta visualization
        self.theta += theta[1:-1]
        self.visualize_theta[...] = np.where(
            visualize_theta[1:-1], 1, self.visualize_theta
        )
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
        theta, visualize_theta = self.apriori_limiter(
            points=np.array([horizontal_points, vertical_points]), u=fallback, m=m, M=M
        )
        # apply smooth extrema detection
        theta = np.where(alpha < 1.0, theta, 1.0)
        # store theta
        stored_theta = chop(theta, self.riemann_zone, axis=0)
        stored_theta = chop(stored_theta, self.riemann_zone, axis=1)
        visualize_theta = chop(visualize_theta, self.riemann_zone, axis=0)
        visualize_theta = chop(visualize_theta, self.riemann_zone, axis=1)
        self.theta += stored_theta
        self.visualize_theta[...] = np.where(visualize_theta, 1, self.visualize_theta)
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
        return np.ones_like(u), np.ones_like(u)

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
            theta   (m,) or (m, n)
        """
        # max and min of u evaluated at quadrature points
        m_ij = np.amin(points, axis=tuple(range(points.ndim - self.ndim)))
        M_ij = np.amax(points, axis=tuple(range(points.ndim - self.ndim)))
        # theta visualization
        u_range = np.max(u) - np.min(u)
        visualize_theta = np.where(
            m_ij - m < -self.visualization_tolerance * u_range, 1, 0
        )
        visualize_theta[...] = np.where(
            M_ij - M > self.visualization_tolerance * u_range, 1, visualize_theta
        )
        # evaluate slope limiter
        theta = np.ones_like(u)
        M_arg = np.abs(M - u) / ((np.abs(M_ij - u)))
        m_arg = np.abs(m - u) / ((np.abs(m_ij - u)))
        theta = np.where(M_arg < theta, M_arg, theta)
        theta = np.where(m_arg < theta, m_arg, theta)
        return theta, visualize_theta

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

        # check for smooth extrema and relax
        if np.any(possible_trouble):
            alpha = self.compute_alpha(self.apply_bc(unew, gw=3, const=self.const))
            trouble = np.where(possible_trouble, np.where(alpha < 1, 1, 0), 0)

        # PAD
        trouble = np.where(unew < self.PAD[0], 1, trouble)
        trouble = np.where(unew > self.PAD[1], 1, trouble)

        # set all cells to 1 if cause_trouble = True
        trouble = (
            1 - self.cause_trouble
        ) * trouble + self.cause_trouble * np.ones_like(self.u0, dtype="int")

        # store history of troubled cells
        self.trouble += trouble
        visualize_trouble = np.where(
            lower_differences < -self.visualization_tolerance * u_range, 1, 0
        )
        visualize_trouble = np.where(
            upper_differences > self.visualization_tolerance * u_range,
            1,
            visualize_trouble,
        )
        self.visualize_trouble[...] = np.where(
            visualize_trouble, 1, self.visualize_trouble
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
        north_face, south_face = compute_fallback_faces(u_2gw[:, 2:-2], axis=0)
        east_face, west_face = compute_fallback_faces(u_2gw[2:-2, :], axis=1)

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

    def check_mpp(self, u):
        tol = 1e-16
        return not np.logical_or(
            np.any(u < self.u0_min - tol), np.any(u > self.u0_max + tol)
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
        approx = self.u[-1]
        truth = self.u[0]
        if norm == "l1":
            return np.sum(np.abs(approx - truth) * self.hx * self.hy)
        if norm == "l2":
            return np.sqrt(np.sum(np.power(approx - truth, 2)) * self.hx * self.hy)
        if norm == "inf":
            return np.max(np.abs(approx - truth))

    def logupdate(self):
        """
        self.t0 has been overwritten with t0 + dt
        self.u0 has been overwritten with the state at t0 + dt
        """
        if self.iteration_count % self.log_every == 0 or self.t0 == self.T:
            na = np.newaxis
            self.t = np.append(self.t, self.t0)
            self.u = np.append(self.u, self.u0[na], axis=0)
            self.theta_history = np.append(
                self.theta_history, self.theta[na] / self.udot_evaluation_count, axis=0
            )
            self.visualize_theta_history = np.append(
                self.visualize_theta_history, self.visualize_theta[na], axis=0
            )
            self.trouble_history = np.append(
                self.trouble_history,
                self.trouble[na] / self.udot_evaluation_count,
                axis=0,
            )
            self.visualize_trouble_history = np.append(
                self.visualize_trouble_history, self.visualize_trouble[na], axis=0
            )
            self.loglen += 1
        # clear theta sum, troubled cell sum, and evaluation count
        self.theta[...] = 0.0
        self.visualize_theta[...] = 0
        self.trouble[...] = 0
        self.visualize_trouble[...] = 0
        self.udot_evaluation_count = 0

    def pre_integrate(self, method_name):
        # create solution path if it doesn't exist
        if not os.path.exists(self._load_directory):
            os.makedirs(self._load_directory)
        self._filename = self._filename + "_" + method_name + ".pkl"
        self.filepath = self._load_directory + self._filename
        # load the solution if it already exists
        if os.path.isfile(self.filepath) and self.load:
            with open(self.filepath, "rb") as thisfile:
                loaded_instance = pickle.load(thisfile)
                self.t = loaded_instance.t
                self.u = loaded_instance.u
                self.theta_history = loaded_instance.theta_history
                self.visualize_theta_history = loaded_instance.visualize_theta_history
                self.trouble_history = loaded_instance.trouble_history
                self.visualize_trouble_history = (
                    loaded_instance.visualize_trouble_history
                )
                self.loglen = loaded_instance.loglen
                self.solution_time = loaded_instance.solution_time
            return False
        # otherwise proceed to integration
        print("New solution instance...")
        return True

    def post_integrate(self):
        # Save the instance to a file
        if self.load:
            with open(self.filepath, "wb") as thisfile:
                pickle.dump(self, thisfile)
            print(f"Wrote a solution instance to {self.filepath}\n")

    def minmax(self):
        if self.ndim == 1:
            minaxis = 1
        elif self.ndim == 2:
            minaxis = (1, 2)
        minimums = np.min(self.u, axis=minaxis) - np.min(self.u[-1])
        maximums = np.max(self.u, axis=minaxis) - np.max(self.u[-1])
        mean_min = np.mean(minimums)
        std_min = np.std(minimums)
        abs_min = np.min(minimums)
        mean_max = np.mean(maximums)
        std_max = np.std(maximums)
        abs_max = np.max(maximums)
        headers = [
            "abs min",
            "mean min",
            "std min",
            "abs max",
            "mean max",
            "std max",
            "time (s)",
        ]
        lines = ["-------"] * 7
        values = [
            abs_min,
            mean_min,
            std_min,
            abs_max,
            mean_max,
            std_max,
            self.solution_time,
        ]
        print("{:>14}{:>14}     {:>11}{:>14}{:>14}     {:>11}{:>14}".format(*headers))
        print("{:>14}{:>14}     {:>11}{:>14}{:>14}     {:>11}{:>14}".format(*lines))
        print(
            "{:14.5e}{:14.5e} +/- {:11.5e}{:14.5e}{:14.5e} +/- {:11.5e}{:14.5f}".format(
                *values
            )
        )
        print()
