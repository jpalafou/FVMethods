"""
defines the AdvectionSolver class, a forward-stepping finite volume solver for
du/dt + df/dx = 0           (1D)
or
du/dt + df/dx + dg/dy = 0   (2D)
where u are cell volume averages and f and g are fluxes in x and y, respectively
"""

import logging
import matplotlib as mpl
import numpy as np
import os
import pickle
from typing import Tuple
from finite_volume.a_priori import mpp_cfl, mpp_limiter
from finite_volume.a_posteriori import (
    broadcast_troubled_cells_to_faces_1d,
    broadcast_troubled_cells_to_faces_2d,
    broadcast_troubled_cells_to_faces_with_blending_1d,
    broadcast_troubled_cells_to_faces_with_blending_2d,
    compute_MUSCL_interpolations_1d,
    compute_MUSCL_interpolations_2d,
    compute_PP2D_interpolations,
    find_trouble,
    minmod,
    moncen,
)
from finite_volume.initial_conditions import generate_ic
from finite_volume.integrate import Integrator
from finite_volume.fvscheme import ConservativeInterpolation, TransverseIntegral
from finite_volume.mathematiques import gauss_lobatto
from finite_volume.riemann import upwinding
from finite_volume.sed import compute_alpha_1d, compute_alpha_2d
from finite_volume.utils import (
    convolve_batch2d,
    pad_uniform_extrap,
    quadrature_mesh,
    RK_dt_adjust,
)


class AdvectionSolver(Integrator):
    """
    args:
        u0:                     initial condition, keywork or callable function
        bc:                     boundary condition type
                                    "periodic"
                                    "dirichlet"
        const:                  constant boundary values for bc = "dirichlet"
                                    None
                                    {"u": u_const, "trouble": trouble_const}
        n:                      number of finite volumes (cells)
                                    float   (nx)        1D grid
                                    tuple   (ny, nx)    2D grid
                                            (nboth,)    square grid
        x:                      tuple of boundaries in x
                                    (xleft, xright)
        y:                      tuple of boundaries in y
                                    None                if 2D grid, x is used
                                    (yleft, yright)
        t0:                     starting time
        snapshot_dt:            dt for snapshots
        num_snapshots:          number of times to evolve system by snapshot_dt
        v:                      advection velocity
                                    float       (v1d)       uniform 1D flow
                                    tuple       (vx, vy)    uniform 2D flow
                                                (vboth,)
                                    callable    v(xx, yy)   2D velocity field
        courant:                CFL factor
                                    float   maximum CFL factor used in the solver
                                    'mpp'   assigns a_priori.mpp_cfl(order)
        order:                  accuracy requirement for spatial discretization
        flux_strategy:          quadrature for integrating fluxes
                                    'gauss-legendre'
                                    'transverse'        cheaper option for 2D grid
        apriori_limiting:       whether to use Zhang and Shu MPP slope limiting
        mpp_lite:               whether to use a variation of apriori_limiting where
                                the cell center is the only interior point
        aposteriori_limiting:   whether to detect troubled cells and revise fluxes with
                                a fallback scheme
        fallback_limiter:       name of fallback slope limiter
                                    'minmod'
                                    'moncen'
                                    'PP2D'      MPP option for 2D grid
        convex:                 whether to smooth the troubled cell revision masks
        hancock:                whether to use a predictor corrector scheme when
                                computing fallback fluxes. if True, the fallback scheme
                                is MUSCL-Hancock.
        fallback_to_1st_order:  whether to fall back again to first-order fluxes in the
                                presence of PAD violations (see PAD).
        cause_trouble:          all cells are always flagged as troubled
        SED:                    whether to disable slope limiting in the presence of
                                smooth extrema detection
        NAD:                    tolerance for numerical admissibility detection in a
                                posteriori limiting
                                    float > 0       implement NAD tolerance
                                    None or np.inf  disable NAD
        PAD:                    physical admissibility detection
                                    (lower_bound, upper_bound)  implement PAD
                                    None or (-np.inf, np.inf)   disable PAD
        adjust_stepsize:        highest-order time integration scheme used when the
                                time step size is adjusted to artificially increase the
                                order of accuracy of the solution to 'order'
                                    None    do not adjust the time step size
                                    int
        adaptive_stepsize:      whether to use an adaptive time step size, halving dt
                                in the precense of an MPP violation. only use this
                                option with an MPP scheme.
        mpp_tolerance:          tolerance added to PAD to avoid triggering the slope
                                limiter in uniform regions
                                    float
                                    None or 0.0     0-tolerance
                                    np.inf          disable PAD
        progress_bar:           whether to print a progress bar in the loop
        load:                   whether to load precalculated solution at
                                save_directory/self._filename
        save:                   whether to write freshly calculated solution to
                                save_directory/self._filename
        save_directory:         directory from which to read/write self._filename
    returns:
        self.snapshots:         list of snapshots [{t: t0, u: u0, ...}, ...]
    """

    def __init__(
        self,
        u0: str = "square",
        bc: str = "periodic",
        const: dict = None,
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
        fallback_limiter: str = "minmod",
        convex: bool = False,
        hancock: bool = False,
        fallback_to_1st_order: bool = False,
        cause_trouble: bool = False,
        SED: bool = False,
        NAD: float = 1e-5,
        PAD: tuple = (0, 1),
        adjust_stepsize: int = None,
        adaptive_stepsize: bool = False,
        mpp_tolerance: float = 1e-10,
        progress_bar: bool = True,
        cupy: bool = False,
        load: bool = True,
        save: bool = True,
        save_directory: str = "data/solutions/",
    ):
        # create filename out of the initialization arguments
        self.load = load
        u0_str = u0.__name__ if callable(u0) else str(u0)
        v_str = v.__name__ if callable(v) else str(v)
        n_for_cupy = n[0] if isinstance(n, (tuple, list)) else n
        if not isinstance(cupy, bool):
            cupy = False if n_for_cupy < cupy else True
        filename_comps = [
            u0_str,
            bc[0],  # use only first character
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
            flux_strategy[0],  # use only first character
            apriori_limiting,
            mpp_lite,
            aposteriori_limiting,
            fallback_limiter[:2],  # use only first two characters
            convex,
            hancock,
            fallback_to_1st_order,
            cause_trouble,
            SED,
            NAD,
            PAD,
            adjust_stepsize,
            adaptive_stepsize,
            mpp_tolerance,
            "cupy" if cupy else "numpy",
        ]
        TF = {True: "T", False: "F"}
        filename_comps = [TF[s] if isinstance(s, bool) else s for s in filename_comps]
        self._filename = "_".join(str(s) for s in filename_comps)
        self._save_directory = save_directory
        self.save = save

        # cupy?
        self.cupy = cupy
        self.xp = np
        if cupy:
            import cupy as cp

            self.xp = cp

        # dimensionality
        if isinstance(n, int):
            self.ndim = 1
        elif isinstance(n, (tuple, list)) and len(n) in {1, 2}:
            self.ndim = 2
        else:
            raise BaseException("n: expected int or tuple of length 1 or 2")

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
            if isinstance(v, (int, float)):
                vx_max, vy_max = abs(v), 0
            else:
                raise BaseException("Expected scalar velocity for 1-dimensional domain")
        if self.ndim == 2:
            if isinstance(v, int):
                raise BaseException("Expected vector velocity for 2-dimensional domain")
            elif isinstance(v, (tuple, list)):  # uniform 2d velocity
                if len(v) == 1:
                    vx_max, vy_max = abs(v[0]), abs(v[0])
                elif len(v) == 2:
                    vx_max, vy_max = abs(v[0]), abs(v[1])
            elif callable(v):  # non-uniform 2d velocity
                # precompute velocity at cell corners and use this as estimate for max
                xx, yy = np.meshgrid(self.x_interface, self.y_interface)
                v_xx_yy = v(xx, yy)
                vx_max, vy_max = np.max(abs(v_xx_yy[0])), np.max(abs(v_xx_yy[1]))
            else:
                raise BaseException("Invalid velocity vector for 2-dimensional domain")

        # time discretization
        self.adjust_stepsize = adjust_stepsize
        self.order = order
        self.p = order - 1
        v_over_h = vx_max / self.hx + vy_max / self.hy
        if v_over_h == 0:
            print("0 velocity case: setting v / h to 0.1")
            v_over_h = 0.1
        self.courant = mpp_cfl(order) if courant == "mpp" else courant
        self.highest = adjust_stepsize
        if adjust_stepsize is not None:
            available = np.array([1, 2, 3, 4, 6])
            if self.highest not in available:
                raise NotImplementedError(f"Order {self.highest} integrator")
            available = available[available <= self.highest]
            sufficient = available[available >= order]
            temporder = max(available) if sufficient.size == 0 else min(sufficient)
            adjusted_courant = courant * min(
                RK_dt_adjust(h=self.hx, spatial_order=order, temporal_order=temporder),
                RK_dt_adjust(h=self.hy, spatial_order=order, temporal_order=temporder),
            )
            if order > temporder and adjusted_courant < courant:
                self.courant = adjusted_courant
                print(f"Reassigned C={self.courant} for order {order}")
        dt = self.courant / v_over_h

        # initial condition
        if isinstance(u0, str):
            u0_arr = generate_ic(type=u0, x=self.x, y=self.y)
        if callable(u0):
            if self.ndim == 1:
                u0_arr = u0(x=self.x)
            if self.ndim == 2:
                u0_arr = u0(x=self.x, y=self.y)

        # initialize simulation/visualized theta and trouble
        self.ones_i = np.ones_like(u0_arr, dtype="int")
        self.ones_f = np.ones_like(u0_arr, dtype="double")
        self.theta = 0.0 * self.ones_f
        self.trouble = 0 * self.ones_i
        self.theta_M_denominator = 0.0 * self.ones_f
        self.theta_m_denominator = 0.0 * self.ones_f
        self.NAD_upper = 0 * self.ones_f
        self.NAD_lower = 0 * self.ones_f

        # initialize timeseries lists
        self.u0_min = np.min(u0_arr)
        self.u0_max = np.max(u0_arr)
        self.all_simulation_times = [t0]
        self.min_history = [self.u0_min]
        self.max_history = [self.u0_max]

        # initialize snapshots
        self.snapshots = [
            {
                "t": t0,
                "u": u0_arr,
                "theta": self.theta + 1.0,
                "trouble": self.trouble.copy(),
                "abs(M_ij - u)": self.theta_M_denominator.copy(),
                "abs(m_ij - u)": self.theta_m_denominator.copy(),
                "unew - M": self.NAD_upper.copy(),
                "m - unew": self.NAD_lower.copy(),
            }
        ]

        # initialize integrator
        super().__init__(
            u0=u0_arr,
            dt=dt,
            snapshot_dt=snapshot_dt,
            num_snapshots=num_snapshots,
            t0=t0,
            progress_bar=progress_bar,
        )

        # boundary conditions for different variables
        variables = ["u", "trouble"]
        if const is not None:
            if not isinstance(const, dict) or not set(const.keys()) == set(variables):
                raise BaseException(f"Invalid const: must have keys {variables}")
        if bc == "periodic":
            self.bc_config = {var: dict(mode="wrap") for var in variables}
        elif bc == "dirichlet":
            self.bc_config = {
                var: dict(constant_values=const[var]) for var in variables
            }
        else:
            ValueError(f"Invalid bc '{bc}'")

        # initialize slope limiting in general
        self.cause_trouble = cause_trouble
        self.udot_evaluation_count = 0

        # initialize a priori slope limiting
        self.apriori_limiting = apriori_limiting
        self.mpp_lite = mpp_lite

        # initialize a posteriori slope limiting
        self.aposteriori_limiting = aposteriori_limiting
        if fallback_limiter in {"minmod", "moncen"}:
            self.slope_limiters = {"minmod": minmod, "moncen": moncen}
            self.fallback_limiter_name = fallback_limiter
        elif fallback_limiter == "PP2D":
            if self.ndim != 2:
                raise TypeError("PP2D limiting is not defined for a 1D solver.")
            if fallback_to_1st_order:
                raise TypeError("PP2D does not fall back to first order.")
            self.fallback_limiter_name = "PP2D"
        else:
            raise ValueError("Invalid slope limiter")
        self.convex = convex
        self.hancock = hancock
        self.fallback_to_1st_order = fallback_to_1st_order

        # SED, NAD, PAD
        self.SED = SED
        self.NAD = np.inf if NAD is None else NAD
        self.PAD = (-np.inf, np.inf) if PAD is None else tuple(sorted(PAD))
        self.mpp_tolerance = 0.0 if mpp_tolerance is None else mpp_tolerance
        self.approximated_maximum_principle = (
            self.PAD[0] - self.mpp_tolerance,
            self.PAD[1] + self.mpp_tolerance,
        )

        # flux reconstruction
        if flux_strategy not in {"gauss-legendre", "transverse"}:
            raise ValueError(f"Invalid flux strategy '{flux_strategy}'")
        if flux_strategy == "transverse" and self.ndim == 1:
            raise TypeError("Transverse flux is not defined for a 1D solver")
        self.flux_strategy = flux_strategy
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

        # define stencil for transverse flux
        if self.flux_strategy == "transverse":
            leg_points = np.array([0.0])
            self.transverse_stencil = TransverseIntegral.construct_from_order(
                self.p + 1
            ).nparray()
            self.transverse_width = len(self.transverse_stencil) // 2

        # define central interpolation for mpp_lite
        if self.mpp_lite:
            self.cell_center_stencil = ConservativeInterpolation.construct_from_order(
                self.p + 1, "center"
            ).nparray(size=conservative_stencil_size)

        # useful stencil counts
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
            # EW interface
            EW_interface_x, EW_interface_y = quadrature_mesh(
                x=self.x_interface, y=self.y, quadrature=leg_points, axis=0
            )
            EW_midpoint_x, EW_midpoint_y = np.meshgrid(self.x_interface, self.y)
            # NS interface
            NS_interface_x, NS_interface_y = quadrature_mesh(
                x=self.x, y=self.y_interface, quadrature=leg_points, axis=1
            )
            NS_midpoint_x, NS_midpoint_y = np.meshgrid(self.x, self.y_interface)
            # evaluate v components normal to cell interfaces
            xx_center, yy_center = np.meshgrid(self.x, self.y)
            if isinstance(v, (tuple, list)):
                vx = v[0]
                self.a = vx * np.ones_like(EW_interface_y)
                self.a_midpoint = vx * np.ones_like(EW_midpoint_x)
                self.a_cell_centers = vx * np.ones_like(xx_center)
                if len(v) == 1:
                    vy = v[0]
                elif len(v) == 2:
                    vy = v[1]
                self.b = vy * np.ones_like(NS_interface_x)
                self.b_midpoint = vy * np.ones_like(NS_midpoint_y)
                self.b_cell_centers = vy * np.ones_like(yy_center)
            elif callable(v):
                self.a = v(EW_interface_x, EW_interface_y)[0]
                self.b = v(NS_interface_x, NS_interface_y)[1]
                self.a_midpoint = v(EW_midpoint_x, EW_midpoint_y)[0]
                self.b_midpoint = v(NS_midpoint_x, NS_midpoint_y)[1]
                self.a_cell_centers = v(xx_center, yy_center)[0]
                self.b_cell_centers = v(xx_center, yy_center)[1]
            if self.flux_strategy == "transverse":
                # extrapolate velocity in boundary
                extended_x_values = pad_uniform_extrap(self.x, self.transverse_width)
                extended_y_values = pad_uniform_extrap(self.y, self.transverse_width)
                EW_interface_x, EW_interface_y = quadrature_mesh(
                    x=self.x_interface,
                    y=extended_y_values,
                    quadrature=leg_points,
                    axis=0,
                )
                NS_interface_x, NS_interface_y = quadrature_mesh(
                    x=extended_x_values,
                    y=self.y_interface,
                    quadrature=leg_points,
                    axis=1,
                )
                if isinstance(v, (tuple, list)):
                    self.a = vx * np.ones_like(EW_interface_x)
                    self.b = vy * np.ones_like(NS_interface_x)
                elif callable(v):
                    self.a = v(EW_interface_x, EW_interface_y)[0]
                    self.b = v(NS_interface_x, NS_interface_y)[1]

        # fluxes
        if self.ndim == 1:
            self.f = np.zeros(nx + 1)
            self.g = np.zeros(nx + 1)
        if self.ndim == 2:
            self.f = np.zeros((ny, nx + 1))  # east/west fluxes
            self.g = np.zeros((ny + 1, nx))  # north/south fluxes

        # function assignment
        self.riemann_solver = upwinding
        if self.ndim == 1:
            self.get_fluxes = self.get_fluxes_1d
            self.compute_alpha = compute_alpha_1d
        elif self.ndim == 2:
            self.get_fluxes = self.get_fluxes_2d
            self.compute_alpha = compute_alpha_2d
        if adaptive_stepsize:
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
        dudt of advection equation dudt + dfdx + dgdy = 0 where f and g are fluxes
        returns:
            dudt:   (nx,) or (ny, nx)
        """
        dfdx = self.hx_recip * (self.f[..., 1:] - self.f[..., :-1])
        dgdy = self.hy_recip * (self.g[1:, ...] - self.g[:-1, ...])
        return -dfdx + -dgdy

    def udot(self, u: np.ndarray, t: float = None, dt: float = None) -> np.ndarray:
        """
        args:
            u:      cell volume averages (nx,) or (ny, nx)
            t:      time at which u is defined
            dt:     timestep size by which to step forward
        returns:
            dudt:   dynamics of cell volume averages (nx,) or (ny, nx)
        """
        self.udot_evaluation_count += 1
        if not (self.cause_trouble and self.aposteriori_limiting):
            self.get_fluxes(u=u)  # high order flux update
        if self.aposteriori_limiting:
            self.aposteriori_limiter(u=u, dt=dt)
        return self.get_dynamics()

    def apply_bc(
        self,
        arr: np.ndarray,
        pad_width: {int, tuple, list},
        mode="u",
    ) -> np.ndarray:
        """
        args:
            arr:        (nx,) or (ny, nx), array without padding
            pad_width:  see documanetation for np.pad
            mode:       name of variable to be np padded
        returns:
            arr with padding
        """
        return np.pad(arr, pad_width=pad_width, **self.bc_config[mode])

    def get_fluxes_1d(self, u: np.ndarray):
        """
        args:
            u:          (nx,)
        overwrites:
            self.f:     (nx + 1,)
        """
        # interpolate points from line averages
        points = convolve_batch2d(
            arr=self.apply_bc(u, pad_width=self.conservative_width + 1).reshape(1, -1),
            kernel=self.pointwise_stencils.reshape(self.n_pointwise_stencils, 1, -1),
            xp=self.xp,
        )
        points = points[0, :, 0, :]  # (# of interpolated points, # cells x
        fallback = self.apply_bc(u, pad_width=1)

        # initialize trivial theta and related terms
        theta = np.ones_like(fallback)
        M_i = fallback
        m_i = fallback

        # a priori limiting
        if self.apriori_limiting:
            mpp_limiting_points = points
            if self.mpp_lite:
                midpoints = self.compute_cell_center(u)
                mpp_limiting_points = np.concatenate((mpp_limiting_points, midpoints))
            theta, M_i, m_i = mpp_limiter(
                u=self.apply_bc(u, pad_width=2),
                points=mpp_limiting_points,
                zeros=self.cause_trouble,
                xp=self.xp,
            )

            # PAD
            PAD = np.logical_or(
                m_i < self.approximated_maximum_principle[0],
                M_i > self.approximated_maximum_principle[1],
            )

            # smooth extrema detection
            alpha = self.compute_alpha(
                self.apply_bc(u, pad_width=4), zeros=not self.SED
            )
            not_smooth_extrema = alpha < 1

            # revise theta with PAD then SED
            theta = np.where(PAD, theta, np.where(not_smooth_extrema, theta, 1.0))

        # log theta and related terms for snapshot
        self.theta += theta[1:-1]
        self.theta_M_denominator += np.abs(M_i[1:-1] - u)
        self.theta_m_denominator += np.abs(m_i[1:-1] - u)

        # limit slopes
        points = theta * (points - fallback) + fallback

        # riemann problem
        right_points = points[-1, :-1]
        left_points = points[0, 1:]
        self.f[...] = self.riemann_solver(
            v=self.a, left_value=right_points, right_value=left_points
        )

    def get_fluxes_2d(self, u: np.ndarray):
        """
        args:
            u:          (ny, nx)
        overwrites:
            self.f:     (ny, nx + 1)
            self.g:     (ny + 1, nx)
        """
        # interpolate line averages from cell averages
        horizontal_lines = convolve_batch2d(
            arr=self.apply_bc(u, pad_width=self.conservative_width + self.riemann_gw)[
                np.newaxis
            ],
            kernel=self.line_stencils.reshape(self.n_line_stencils, -1, 1),
            xp=self.xp,
        )
        vertical_lines = convolve_batch2d(
            arr=self.apply_bc(u, pad_width=self.conservative_width + self.riemann_gw)[
                np.newaxis
            ],
            kernel=self.line_stencils.reshape(self.n_line_stencils, 1, -1),
            xp=self.xp,
        )

        # interpolate points from line averages
        horizontal_points = convolve_batch2d(
            arr=horizontal_lines[0],
            kernel=self.pointwise_stencils.reshape(self.n_pointwise_stencils, 1, -1),
            xp=self.xp,
        )
        vertical_points = convolve_batch2d(
            arr=vertical_lines[0],
            kernel=self.pointwise_stencils.reshape(self.n_pointwise_stencils, -1, 1),
            xp=self.xp,
        )
        fallback = self.apply_bc(u, pad_width=self.riemann_gw)

        # initialize trivial theta and related terms
        theta = np.ones_like(fallback)
        M_ij = fallback
        m_ij = fallback

        # a priori slope limiting
        if self.apriori_limiting:
            h_shp = horizontal_points.shape
            hps = horizontal_points.reshape(h_shp[0] * h_shp[1], h_shp[2], h_shp[3])
            v_shp = vertical_points.shape
            vps = vertical_points.reshape(v_shp[0] * v_shp[1], v_shp[2], v_shp[3])
            mpp_limiting_points = np.concatenate((hps, vps))
            if self.mpp_lite:
                midpoints = self.compute_cell_center(u)
                mpp_limiting_points = np.concatenate((mpp_limiting_points, midpoints))
            theta, M_ij, m_ij = mpp_limiter(
                u=self.apply_bc(u, pad_width=self.riemann_gw + 1),
                points=mpp_limiting_points,
                zeros=self.cause_trouble,
                xp=self.xp,
            )

            # PAD
            PAD = np.logical_or(
                m_ij < self.approximated_maximum_principle[0],
                M_ij > self.approximated_maximum_principle[1],
            )

            # smooth extrema detection
            alpha = self.compute_alpha(
                self.apply_bc(u, pad_width=self.riemann_gw + 3), zeros=not self.SED
            )
            not_smooth_extrema = alpha < 1

            # revise theta with PAD then SED
            theta = np.where(PAD, theta, np.where(not_smooth_extrema, theta, 1.0))

        # limit slopes
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

        # log theta and related terms for snapshot
        self.theta += theta[riemann_slice, riemann_slice][1:-1, 1:-1]
        self.theta_M_denominator += np.abs(
            M_ij[riemann_slice, riemann_slice][1:-1, 1:-1] - u
        )
        self.theta_m_denominator += np.abs(
            m_ij[riemann_slice, riemann_slice][1:-1, 1:-1] - u
        )

        # riemann solver
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
            u:              cell volume averages with padding
                            (nx + p,) or (ny + p, nx + q)
        returns:
            midpoints:      cell midpoints  (nx,) or (ny, nx)
        """
        if self.ndim == 1:
            midpoints = convolve_batch2d(
                arr=self.apply_bc(u, pad_width=self.conservative_width + 1)[np.newaxis],
                kernel=self.cell_center_stencil.reshape(1, -1),
                xp=self.xp,
            )
            return midpoints[0, 0, ...]
        elif self.ndim == 2:
            horizontal_lines = convolve_batch2d(
                arr=self.apply_bc(
                    u, pad_width=self.conservative_width + self.riemann_gw
                )[np.newaxis],
                kernel=self.cell_center_stencil.reshape(1, -1, 1),
                xp=self.xp,
            )
            midpoints = convolve_batch2d(
                arr=horizontal_lines[0],
                kernel=self.cell_center_stencil.reshape(1, 1, -1),
                xp=self.xp,
            )
            return midpoints[0, ...]

    def find_trouble(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        args:
            u:          cell volume averages (nx,) or (ny, nx)
            dt:         time step size
        returns:
            trouble:    boolean array (nx,) or (ny, nx)
        """
        unew = u + dt * self.get_dynamics()
        trouble, M, m = find_trouble(
            u=self.apply_bc(u, pad_width=1),
            u_candidate=self.apply_bc(unew, pad_width=3),
            NAD=self.NAD,
            PAD=self.approximated_maximum_principle,
            SED=self.SED,
            ones=self.cause_trouble,
            xp=self.xp,
        )
        # log troubled cells and related terms for snapshot
        self.trouble += trouble
        self.NAD_upper += unew - M
        self.NAD_lower += m - unew
        return trouble

    def revise_fluxes_1d(self, u: np.ndarray, dt: float):
        """
        args:
            u:          cell volume averages (nx,)
            dt:         time step size
        overwrites:
            self.f:     (nx + 1,)
        """
        # compute fallback fluxes
        left_fallback_face, right_fallback_face = compute_MUSCL_interpolations_1d(
            u=self.apply_bc(u, pad_width=1),
            slope_limiter=self.slope_limiters[self.fallback_limiter_name],
            fallback_to_1st_order=self.fallback_to_1st_order,
            PAD=self.approximated_maximum_principle,
            hancock=self.hancock,
            dt=dt,
            h=self.hx,
            v_cell_centers=self.a_cell_centers,
        )
        left_fallback_face = self.apply_bc(left_fallback_face, pad_width=1)
        right_fallback_face = self.apply_bc(right_fallback_face, pad_width=1)
        fallback_fluxes = self.riemann_solver(
            v=self.a,
            left_value=right_fallback_face[:-1],
            right_value=left_fallback_face[1:],
        )
        # find troubled cells
        trouble = self.find_trouble(u, dt)
        # revise fluxes
        if not self.convex:
            troubled_interface_mask = broadcast_troubled_cells_to_faces_1d(
                trouble=trouble, xp=self.xp
            )
        else:
            troubled_interface_mask = (
                broadcast_troubled_cells_to_faces_with_blending_1d(
                    trouble=self.apply_bc(trouble, pad_width=2, mode="trouble"),
                    xp=self.xp,
                )
            )
        self.f[...] = (
            1 - troubled_interface_mask
        ) * self.f + troubled_interface_mask * fallback_fluxes

    def revise_fluxes_2d(self, u: np.ndarray, dt: float):
        """
        args:
            u:          cell volume averages (ny, nx)
            dt:         time step size
        overwrites:
            self.f:     (ny, nx + 1)
            self.g:     (ny + 1, nx)
        """
        # compute fallback fluxes
        if self.fallback_limiter_name in {"minmod", "moncen"}:
            fallback_faces = compute_MUSCL_interpolations_2d(
                u=self.apply_bc(u, pad_width=1),
                slope_limiter=self.slope_limiters[self.fallback_limiter_name],
                fallback_to_1st_order=self.fallback_to_1st_order,
                PAD=self.approximated_maximum_principle,
                hancock=self.hancock,
                dt=self.dt,
                h=(self.hx, self.hy),
                v_cell_centers=(self.a_cell_centers, self.b_cell_centers),
            )
        elif self.fallback_limiter_name == "PP2D":
            fallback_faces = compute_PP2D_interpolations(
                u=self.apply_bc(u, pad_width=1),
                hancock=self.hancock,
                dt=self.dt,
                h=(self.hx, self.hy),
                v_cell_centers=(self.a_cell_centers, self.b_cell_centers),
                xp=self.xp,
            )
        north_face = self.apply_bc(fallback_faces[1][1], pad_width=1)
        south_face = self.apply_bc(fallback_faces[1][0], pad_width=1)
        east_face = self.apply_bc(fallback_faces[0][1], pad_width=1)
        west_face = self.apply_bc(fallback_faces[0][0], pad_width=1)
        fallback_fluxes_x = self.riemann_solver(
            v=self.a_midpoint,
            left_value=east_face[1:-1, :-1],
            right_value=west_face[1:-1, 1:],
        )
        fallback_fluxes_y = self.riemann_solver(
            v=self.b_midpoint,
            left_value=north_face[:-1, 1:-1],
            right_value=south_face[1:, 1:-1],
        )
        # find troubled cells
        trouble = self.find_trouble(u, dt)
        # overwrite fluxes
        if not self.convex:
            (
                troubled_interface_mask_x,
                troubled_interface_mask_y,
            ) = broadcast_troubled_cells_to_faces_2d(trouble=trouble, xp=self.xp)
        else:
            (
                troubled_interface_mask_x,
                troubled_interface_mask_y,
            ) = broadcast_troubled_cells_to_faces_with_blending_2d(
                trouble=self.apply_bc(trouble, pad_width=2, mode="trouble"), xp=self.xp
            )
        self.f[...] = (
            1 - troubled_interface_mask_x
        ) * self.f + troubled_interface_mask_x * fallback_fluxes_x
        self.g[...] = (
            1 - troubled_interface_mask_y
        ) * self.g + troubled_interface_mask_y * fallback_fluxes_y

    def gauss_legendre_integral(
        self, pointwise_fluxes: np.ndarray, axis: int
    ) -> np.ndarray:
        """
        args:
            pointwise_fluxes:   (p, m, n)
        returns:
            out:                (1, 1, m, n)
        """
        na = np.newaxis
        return np.sum(pointwise_fluxes * self.leg_weights, axis=0)[na, na]

    def transverse_integral(
        self, pointwise_fluxes: np.ndarray, axis: int
    ) -> np.ndarray:
        """
        args:
            pointwise_fluxes:   (1, m + excess, n) or (1, m, n + excess)
        returns:
            out:                (1, 1, m, n)
        """
        kernel_shape = [1, 1, 1]
        kernel_shape[axis + 1] = len(self.transverse_stencil)
        return convolve_batch2d(
            arr=pointwise_fluxes,
            kernel=self.transverse_stencil.reshape(kernel_shape),
            xp=self.xp,
        )

    def rkorder(self, ssp: bool = True, rk6: bool = False):
        """
        rk integrate to an order that matches the spatial order
        args:
            ssp:        whether to use strong-stability-preserving time integration
            rk6:        whether to enable 6th order Runge-Kutta time integration
        """
        if self.highest == 6 and not rk6:
            raise BaseException("Set rk6=True or choose a different adjust_stepsize")
        if self.order == 1 or self.highest == 1:
            self.euler()
        elif self.order == 2 or self.highest == 2:
            if ssp:
                self.ssprk2()
            else:
                self.rk2()
        elif self.order == 3 or self.highest == 3:
            if ssp:
                self.ssprk3()
            else:
                self.rk3()
        elif self.order == 4 or self.highest == 4 or not rk6:
            self.rk4()
        elif rk6:
            self.rk6()

    def periodic_error(self, norm: str = "l1"):
        """
        args:
            norm:   'l1', 'l2', or 'linf'
        returns:
            out:    norm specified error (nx,) or (ny, nx)
        """
        approx = self.snapshots[-1]["u"]
        truth = self.snapshots[0]["u"]
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
        self.min_history.append(np.min(self.u0).item())
        self.max_history.append(np.max(self.u0).item())

    def snapshot(self):
        if self.cupy:
            self.send_to_CPU()
        self.snapshots.append(
            {
                "t": self.t0,
                "u": self.u0,
                "theta": self.theta / self.udot_evaluation_count,
                "trouble": self.trouble / self.udot_evaluation_count,
                "abs(M_ij - u)": self.theta_M_denominator / self.udot_evaluation_count,
                "abs(m_ij - u)": self.theta_m_denominator / self.udot_evaluation_count,
                "unew - M": self.NAD_upper / self.udot_evaluation_count,
                "m - unew": self.NAD_lower / self.udot_evaluation_count,
            }
        )
        if self.cupy:
            self.send_to_GPU()

    def step_cleanup(self):
        """
        reset values of attributes which accumulate over substeps
        """
        # clear theta sum, troubled cell sum, and evaluation count
        self.theta[...] = 0.0
        self.trouble[...] = 0
        self.theta_M_denominator[...] = 0.0
        self.theta_m_denominator[...] = 0.0
        self.NAD_upper[...] = 0.0
        self.NAD_lower[...] = 0.0
        self.udot_evaluation_count = 0
        self.append_to_timeseries_lists()

    def refine_timestep(self, dt):
        return dt / 2

    def pre_integrate(self, method_name):
        """
        Attempt to load solution from self.filepath. If self.filepath does not exist
        or loading results in any error, compute a fresh solution.
        """
        # find filepath where solution is/will be stored
        self._filename = self._filename + "_" + method_name + ".pkl"
        self.filepath = self._save_directory + self._filename
        # load the solution if it already exists
        if os.path.isfile(self.filepath) and self.load:
            try:
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
            except Exception as e:
                logging.warning(f"Failed to load {self.filepath}: {e}")
        # otherwise proceed to integration
        try:
            os.makedirs(self._save_directory)
        except OSError:
            pass
        print("New solution instance...")
        # send data to GPU
        if self.cupy:
            self.send_to_GPU()
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
        # send data to CPU
        if self.cupy:
            self.send_to_CPU()
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

    def report_mpp_violations(self):
        _, stats = self.compute_mpp_violations()

        headers = [
            "",
            "worst",
            "frequency",
            "mean",
        ]

        upper_values = ["upper", stats["worst upper"], stats["upper frequency"], ""]
        lower_values = ["lower", stats["worst lower"], stats["lower frequency"], ""]
        total_values = ["total", stats["worst"], stats["frequency"], stats["mean"]]

        print("\n{:>14}{:>14}{:>14}{:>14}".format(*headers))
        print("{:>14}{:14.5e}{:14.5e}{:>14}".format(*upper_values))
        print("{:>14}{:14.5e}{:14.5e}{:>14}".format(*lower_values))
        print("{:>14}{:14.5e}{:14.5e}{:14.5e}\n".format(*total_values))

    def compute_mpp_violations(self) -> Tuple[np.ndarray, dict]:
        mins = np.array(self.min_history)
        maxs = np.array(self.max_history)
        lower_bound_violations = mins - self.PAD[0]
        upper_bound_violations = self.PAD[1] - maxs
        violations = np.minimum(lower_bound_violations, upper_bound_violations)
        # negative values indicate violation
        stats = {}
        stats["worst"] = np.min(violations)
        stats["worst upper"] = np.min(upper_bound_violations)
        stats["worst lower"] = np.min(lower_bound_violations)
        stats["mean"] = np.mean(violations)
        stats["standard deviation"] = np.std(violations)
        stats["lower frequency"] = np.sum(
            np.where(lower_bound_violations < 0, 1, 0)
        ) / (self.step_count + 1)
        stats["upper frequency"] = np.sum(
            np.where(upper_bound_violations < 0, 1, 0)
        ) / (self.step_count + 1)
        stats["frequency"] = np.sum(np.where(violations < 0, 1, 0)) / (
            self.step_count + 1
        )
        return violations, stats

    def redefine_arrays(self, xp_asarray: callable):
        """
        redefine arrays as xp arrays
        args:
            xp_asarray:     cp.asnumpy or cp.asarray
        """

        # snapshot
        self.u0 = xp_asarray(self.u0)
        self.theta = xp_asarray(self.theta)
        self.trouble = xp_asarray(self.trouble)
        self.theta_M_denominator = xp_asarray(self.theta_M_denominator)
        self.theta_m_denominator = xp_asarray(self.theta_m_denominator)
        self.NAD_upper = xp_asarray(self.NAD_upper)
        self.NAD_lower = xp_asarray(self.NAD_lower)

        # velocities and fluxes
        self.a = xp_asarray(self.a)
        self.a_cell_centers = xp_asarray(self.a_cell_centers)
        self.f = xp_asarray(self.f)
        self.g = xp_asarray(self.g)
        if self.ndim == 2:
            self.b = xp_asarray(self.b)
            self.a_midpoint = xp_asarray(self.a_midpoint)
            self.b_midpoint = xp_asarray(self.b_midpoint)
            self.b_cell_centers = xp_asarray(self.b_cell_centers)

        # quadrature
        self.line_stencils = xp_asarray(self.line_stencils)
        self.pointwise_stencils = xp_asarray(self.pointwise_stencils)
        if self.flux_strategy == "gauss-legendre":
            self.leg_weights = xp_asarray(self.leg_weights)
        elif self.flux_strategy == "transverse":
            self.transverse_stencil = xp_asarray(self.transverse_stencil)

        # slope limiting
        if self.mpp_lite:
            self.cell_center_stencil = xp_asarray(self.cell_center_stencil)

    def send_to_CPU(self):
        """
        redefine all attribute arrays as numpy arrays
        """
        self.redefine_arrays(xp_asarray=self.xp.asnumpy)

    def send_to_GPU(self):
        """
        redefine all attribute arrays as cupy arrays
        """
        self.redefine_arrays(xp_asarray=self.xp.asarray)

    def __getstate__(self):
        """
        when pickling, remove xp attribute from state dictionary
        """
        state = self.__dict__.copy()
        del state["xp"]
        return state

    def __setstate__(self, state):
        """
        restore xp attribute after pickling
        """
        self.__dict__.update(state)
        self.xp = np
        if self.cupy:
            import cupy as cp

            self.xp = cp

    def _hide_small_violations(
        self, i: int, spatial_slices: tuple, mode: str, tolerance: float
    ) -> np.ndarray:
        """
        args:
            i:                  snapshot index
            spatial_slices:     tuple of slices
            mode:               "theta"     ignore small abs(M_ij - u) & abs(m_ij - u)
                                "trouble"   ignore small unew - M & m - unews
            tolerance:          float
                                -np.inf     return all violations
        returns:
            out:                theta or trouble where small violations are ignored
        """
        if len(spatial_slices) != self.ndim:
            raise BaseException(f"Invalid array slicing for ndim={self.ndim}")
        if mode == "theta":
            ydata = self.snapshots[i]["theta"][spatial_slices]
            hide_small_violations = np.logical_and(
                self.snapshots[i]["abs(M_ij - u)"][spatial_slices] < tolerance,
                self.snapshots[i]["abs(m_ij - u)"][spatial_slices] < tolerance,
            )
            out = np.where(hide_small_violations, 1, ydata)
        elif mode == "trouble":
            ydata = self.snapshots[i]["trouble"][spatial_slices]
            hide_small_violations = np.logical_and(
                self.snapshots[i]["unew - M"][spatial_slices] < tolerance,
                self.snapshots[i]["m - unew"][spatial_slices] < tolerance,
            )
            out = np.where(hide_small_violations, 0, ydata)
        return out

    def plot_slice(
        self,
        ax,
        i: int = -1,
        x: float = None,
        y: float = None,
        mode: str = "u",
        tolerance: float = 1e-5,
        show_messages: bool = False,
        **kwargs,
    ):
        """
        args:
            ax:             Axes instance from matplotlib
            i:              snapshot index
            x:              x-value to slice along. chooses closest.
            y:              y-value to slice along. chooses closest.
            mode:           variable to plot
                                "u"
                                "theta"     plots 1 - theta
                                "trouble"
            tolerance:      float       ignore small violations of theta or trouble
                            -np.inf     plot all violations of theta or trouble
            show_messages:  whether to print helpful messages
            kwargs:         plt.plot(**kwargs)
        returns:
            see matplotlib.pyplot.plot()
        """
        if self.ndim == 1:
            xdata = self.x
            spatial_slices = (slice(None),)
        elif self.ndim == 2:
            if (x is None and y is None) or (x is not None and y is not None):
                raise BaseException("Provide one value for x or y")
            if x is None:
                xdata = self.x
                j = np.abs(self.y - y).argmin()
                spatial_slices = (j, slice(None))
                if show_messages:
                    print(f"y={self.y[j]}")
            elif y is None:
                xdata = self.y
                j = np.abs(self.x - x).argmin()
                spatial_slices = (slice(None), j)
                if show_messages:
                    print(f"x={self.x[j]}")
        if mode == "u":
            ydata = self.snapshots[i][mode][spatial_slices]
        elif mode in {"theta", "trouble"}:
            ydata = self._hide_small_violations(
                i=i, spatial_slices=spatial_slices, mode=mode, tolerance=tolerance
            )
        if mode == "theta":
            ydata = 1 - ydata
        return ax.plot(xdata, ydata, **kwargs)

    def plot_map(
        self, ax, i: int = -1, mode: str = "u", tolerance: float = 1e-5, **kwargs
    ):
        """
        args:
            ax:             Axes instance from matplotlib
            i:              snapshot index
            mode:           variable to plot
                                "u"
                                "theta"     plots 1 - theta
                                "trouble"
            tolerance:      float       ignore small violations of theta or trouble
                            -np.inf     plot all violations of theta or trouble
            kwargs:         plt.imshow(**kwargs)
        returns:
            see matplotlib.pyplot.imshow()
        """
        if self.ndim != 2:
            raise BaseException("Map plot only defined for ndim=2")
        if mode == "u":
            zdata = self.snapshots[i][mode]
            vmin, vmax = self.PAD
        elif mode in {"theta", "trouble"}:
            zdata = self._hide_small_violations(
                i=i,
                spatial_slices=(slice(None), slice(None)),
                mode=mode,
                tolerance=tolerance,
            )
            vmin, vmax = 0, 1
        if mode == "theta":
            zdata = 1 - zdata
        limits = (self.x[0], self.x[-1], self.y[0], self.y[-1])
        zdata = np.flipud(zdata)
        return ax.imshow(zdata, extent=limits, **kwargs, vmin=vmin, vmax=vmax)

    def plot_cubes(
        self,
        ax,
        i: int = -1,
        xlims: tuple = None,
        ylims: tuple = None,
        edgecolor: str = "black",
        linewidth: float = 0.05,
        azdeg: float = 45,
        altdeg: float = 45,
        raise_floor: bool = False,
        zoom: float = 0.8,
        **kwargs,
    ):
        """
        args:
            ax:             Axes instance from matplotlib
            i:              snapshot index
            xlims:          (lower, upper) constrains x-direction of plotting region
            ylims:          (lower, upper) constrains y-direction of plotting region
            edgecolor:      color of the edge of each cube
            linewidth:      width of edges
            azdeg:          the azimuth (0-360, degrees clockwise from +y-axis) of the
                            light source
            altdeg:         the altitude (0-90, degrees up from the x-y plane) of the
                            light source
            raise_floor:    avoid undershoots of the 3d projection floor
            zoom:           zoom-out factor (choose 1 for no zoom-out)
            kwargs:         ax.bar3d(**kwargs)
        returns:
            see matplotlib.pyplot.imshow()
        """
        if self.ndim == 1:
            raise TypeError("Cannot generate cube plot for 1D solution")

        # set up x and y
        X, Y = np.meshgrid(self.x, self.y)
        xlims = (-np.inf, np.inf) if xlims is None else xlims
        ylims = (-np.inf, np.inf) if ylims is None else ylims
        idxs = ((X >= xlims[0]) & (X <= xlims[1])) & ((Y >= ylims[0]) & (Y <= ylims[1]))
        X, Y = X[idxs], Y[idxs]

        # set up z
        Z = self.snapshots[i]["u"][idxs]
        floor = np.min(Z)

        # set up lightsource and plot
        lightsource = mpl.colors.LightSource(azdeg=azdeg, altdeg=altdeg)
        ax.bar3d(
            x=X,
            y=Y,
            z=floor if raise_floor else 0,
            dx=self.hx,
            dy=self.hy,
            dz=Z - floor if raise_floor else Z,
            lightsource=lightsource,
            edgecolor=edgecolor,
            linewidth=linewidth,
            **kwargs,
        )
        ax.set_box_aspect(aspect=None, zoom=zoom)
