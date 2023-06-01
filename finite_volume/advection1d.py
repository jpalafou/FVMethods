# advection solution schems
import numpy as np
from finite_volume.initial_conditions import generate_ic
from finite_volume.integrate import Integrator
from finite_volume.fvscheme import ConservativeInterpolation
from finite_volume.aposteriori.simple_trouble_detection1d import (
    trouble_detection1d,
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


# class definition
class AdvectionSolver(Integrator):
    """
    args:
        u0  preset string or callable function describing solution at t=0
        n:  number of cells
        x:  tuple of boundaries for x domain
        T:  solving time
        a:  constant advection speed
        courant:    stability condition
        order:  accuracy requirement for polynomial interpolation
        bc:     string describing a pre-coded boudnary condition
        apriori_limiting:   whether to apply an a priori slope limiter
                            can be None, mpp, or mpp lite
        smooth_extrema_detection:   whether to detect smooth extrema
        aposteriori_limiting:   whether to apply a fallback scheme
        loglen: number of saved states
        adujst_time_step:   whether to reduce timestep for order >4
    returns:
        u   array of saved states
    """

    def __init__(
        self,
        u0: str = "square",
        n: int = 32,
        x: tuple = (0, 1),
        T: float = 1,
        a: float = 1,
        courant: float = 0.5,
        order: int = 1,
        bc: str = "periodic",
        apriori_limiting: str = None,
        smooth_extrema_detection: bool = False,
        aposteriori_limiting: bool = False,
        loglen: int = 11,
        adujst_time_step: bool = False,
    ):
        # misc. attributes
        self.order = order
        self.apriori_limiting = apriori_limiting
        self.smooth_extrema_detection = smooth_extrema_detection
        self.aposteriori_limiting = aposteriori_limiting
        self.adujst_time_step = adujst_time_step
        if apriori_limiting not in (None, "mpp", "mpp lite"):
            raise BaseException(f"Invalid apriori limiting type: {apriori_limiting}")

        # spatial discretization
        self.n = n
        self.x_interface = np.linspace(x[0], x[1], num=n + 1)  # cell interfaces
        self.x = 0.5 * (self.x_interface[:-1] + self.x_interface[1:])  # cell centers
        self.L = x[1] - x[0]  # domain size
        self.h = self.L / n  # cell size

        # constant advection velocity defined at cell interfaces
        self.a = a * np.ones(len(self.x_interface))

        # time discretization
        self.courant = courant
        Dt = courant * self.h / max(np.abs(a), 1e-6)
        Dt_adjustment = 1
        if self.adujst_time_step and order > 4:
            Dt_adjustment = rk4_Dt_adjust(self.h, self.L, order)
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
        if isinstance(u0, str):
            u0 = generate_ic(type=u0, x=self.x)
        self.bc = bc

        # initialize integrator
        super().__init__(u0=u0, t=self.t, loglen=loglen)

        # interpolating values at cell interfaces
        left_interface_stensil = ConservativeInterpolation.construct_from_order(
            order, "left"
        ).nparray()
        right_interface_stensil = ConservativeInterpolation.construct_from_order(
            order, "right"
        ).nparray()
        self._stensil_size = len(left_interface_stensil)  # assume symmetric stensils
        k = int(np.floor(len(left_interface_stensil) / 2))  # length of stensil arms
        self.gw = k + 1

        # interpolating values inside cells
        p = order - 1  # polynomial degree p
        q = int(np.ceil((p + 1) / 2)) + 1  # of required quadrature points
        interior_stensils = []
        if apriori_limiting and q > 2:
            if apriori_limiting == "mpp lite":
                # lite version of mpp uses only the cell center as a free absicca
                cell_center_stensil = ConservativeInterpolation.construct_from_order(
                    order, "center"
                ).nparray()
                interior_stensils.append(cell_center_stensil)
            elif apriori_limiting == "mpp":
                # full version requires a stensil for each Guass-Lobatto point
                free_abscissas, _ = np.polynomial.legendre.leggauss(q - 2)
                # transform to cell coordinate
                free_abscissas /= 2
                for x in free_abscissas:
                    stensil = ConservativeInterpolation.construct_from_order(
                        order, x
                    ).nparray()
                    # if the stensil is short, assume it needs a 0 on either end
                    while len(stensil) < self._stensil_size:
                        stensil = np.concatenate((np.zeros(1), stensil, np.zeros(1)))
                    assert len(stensil) == self._stensil_size
                    interior_stensils.append(stensil)
            # quadrature weight at endpoints
            # https://mathworld.wolfram.com/LobattoQuadrature.html
            endpoint_weight = 2 / (q * (q - 1))
            # reduced Courant condition for monotonicity (transform to cell coordinate)
            C_for_mpp = endpoint_weight / 2
            # check if our solution will satisfy mpp
            if abs(a) * self.Dt / self.h >= C_for_mpp:
                print(
                    "WARNING: Maximum principle preserving not satisfied.\nTry a",
                    f" timestep less than {C_for_mpp * self.h / abs(a)} for a",
                    f" Courant condition of {C_for_mpp}.",
                )

        # complete list of interpolation stensils going from left to right
        self.list_of_stensils = (
            [left_interface_stensil] + interior_stensils + [right_interface_stensil]
        )

        # array to store interpolations at each cell
        self.left_face_values = np.zeros(n + 2)
        self.right_face_values = np.zeros(n + 2)

    def apply_bc(self, u_without_ghost_cells: np.ndarray, gw: int) -> np.ndarray:
        """
        args:
            u_without_ghost_cells   1d np array
            gw      number of ghost cells on either side of u
        returns:
            u_with_ghost_cells
        """
        negatvie_gw = -gw
        if self.bc == "periodic":
            left_ghost_zone = u_without_ghost_cells[negatvie_gw:]
            right_ghost_zone = u_without_ghost_cells[:gw]
            u_with_ghost_cells = np.concatenate(
                (left_ghost_zone, u_without_ghost_cells, right_ghost_zone)
            )
        return u_with_ghost_cells

    def riemann(
        self,
        velocities_at_boundaries: float,
        left_of_boundary_values: float,
        right_of_boundary_values: float,
    ) -> float:
        fluxes = np.zeros(len(velocities_at_boundaries))
        fluxes = np.where(velocities_at_boundaries > 0, left_of_boundary_values, fluxes)
        fluxes = np.where(
            velocities_at_boundaries < 0, right_of_boundary_values, fluxes
        )
        return fluxes * velocities_at_boundaries

    def detect_smooth_extrema(self, u: np.ndarray) -> np.ndarray:
        dudx = (u[2:] - u[:-2]) / (2 * self.h)
        left_difference = dudx[1:-1] - dudx[:-2]
        right_difference = dudx[2:] - dudx[1:-1]
        central_difference = 0.5 * (left_difference + right_difference)
        alphaL = np.zeros(len(central_difference))
        alphaL = np.where(
            central_difference < 0,
            np.minimum(np.minimum(2 * left_difference, 0) / central_difference, 1),
            alphaL,
        )
        alphaL = np.where(
            central_difference > 0,
            np.minimum(np.maximum(2 * left_difference, 0) / central_difference, 1),
            alphaL,
        )
        alphaR = np.zeros(len(central_difference))
        alphaR = np.where(
            central_difference < 0,
            np.minimum(np.minimum(2 * right_difference, 0) / central_difference, 1),
            alphaR,
        )
        alphaR = np.where(
            central_difference > 0,
            np.minimum(np.maximum(2 * right_difference, 0) / central_difference, 1),
            alphaR,
        )
        alpha = np.minimum(alphaL, alphaR)
        apply_here = np.where(
            np.amin(np.array([alpha[2:], alpha[1:-1], alpha[:-2]]), axis=0) < 1,
            1,
            0,
        )
        assert len(apply_here) == len(u) - 6
        return apply_here

    def udot(self, u: np.ndarray, t_i: float, dt: float = None) -> np.ndarray:
        ubar_extended = self.apply_bc(u, self.gw)
        n, nwg = len(u), len(ubar_extended)
        # construct an array of staggered state vectors such that a matrix operation
        # with a stensil multiplies weights by their referenced cells
        list_of_windows = []
        for i in range(self._stensil_size):
            right_ind = i + nwg - (self._stensil_size - 1)
            list_of_windows.append(ubar_extended[i:right_ind])
        array_of_windows = np.array(list_of_windows).T
        # interpolate each x value in each cell
        list_of_interpolations = []
        for stensil in self.list_of_stensils:
            list_of_interpolations.append(array_of_windows @ stensil / sum(stensil))
        interpolations = np.array(list_of_interpolations)
        # slope limiter
        theta_i = np.ones(n + 2)
        ubar_1gw = self.apply_bc(u, 1)
        if self.apriori_limiting:
            ubar_2gw = self.apply_bc(u, 2)
            M = np.maximum(ubar_2gw[:-2], np.maximum(ubar_2gw[1:-1], ubar_2gw[2:]))
            m = np.minimum(ubar_2gw[:-2], np.minimum(ubar_2gw[1:-1], ubar_2gw[2:]))
            M_i = np.amax(interpolations, axis=0)
            m_i = np.amin(interpolations, axis=0)
            theta_arg_M = np.abs(M - ubar_1gw) / (np.abs(M_i - ubar_1gw) + 1e-6)
            theta_arg_m = np.abs(m - ubar_1gw) / (np.abs(m_i - ubar_1gw) + 1e-6)
            theta_i = np.where(theta_arg_M < theta_i, theta_arg_M, theta_i)
            theta_i = np.where(theta_arg_m < theta_i, theta_arg_m, theta_i)
            if self.smooth_extrema_detection:
                ubar_4gw = self.apply_bc(u, 4)
                theta_i = np.where(self.detect_smooth_extrema(ubar_4gw), theta_i, 1)
        # slope limited interpolation at cell faces
        self.left_face_values = (
            theta_i * (list_of_interpolations[0] - ubar_1gw) + ubar_1gw
        )
        self.right_face_values = (
            theta_i * (list_of_interpolations[-1] - ubar_1gw) + ubar_1gw
        )
        # flux = a * dU
        self.fluxes = self.riemann(
            self.a, self.right_face_values[:-1], self.left_face_values[1:]
        )
        if self.aposteriori_limiting:
            unew = u + dt * -(self.fluxes[1:] - self.fluxes[:-1]) / self.h
            self.revise_solution(u0=u, ucandidate=unew, t_i=t_i)
        return -(self.fluxes[1:] - self.fluxes[:-1]) / self.h

    def revise_solution(
        self, u0: np.ndarray, ucandidate: np.ndarray, t_i: float
    ) -> np.ndarray:
        # give the previous and candidate solutions two ghost cells
        u0_2gw = self.apply_bc(u0, 2)
        unew_2gw = self.apply_bc(ucandidate, 2)
        # reshape into 3d array to match david's code
        u0_2gw = u0_2gw.reshape(1, 1, -1)
        unew_2gw = unew_2gw.reshape(1, 1, -1)
        # find troubled cells
        troubled_cells = trouble_detection1d(u0_2gw, unew_2gw, self.h)
        troubled_faces = np.zeros((1, 1, len(u0) + 1))
        # find troubled faces
        if np.any(troubled_cells):
            troubled_faces[:, :, :-1] = troubled_cells
            troubled_faces[:, :, 1:] = np.where(
                troubled_cells == 1, 1, troubled_faces[:, :, 1:]
            )
            # find 2nd order face values
            (
                order2_left_interpolation,
                order2_right_interpolation,
            ) = compute_second_order_fluxes(u0_2gw)
            # replace troubled faces with second order interpolations
            self.left_face_values[1:] = np.where(
                troubled_faces[0, 0, :],
                order2_left_interpolation[0, 0, 1:],
                self.left_face_values[1:],
            )
            self.right_face_values[:-1] = np.where(
                troubled_faces[0, 0, :],
                order2_right_interpolation[0, 0, :-1],
                self.right_face_values[:-1],
            )
            # revise fluxes
            self.fluxes = self.riemann(
                self.a, self.right_face_values[:-1], self.left_face_values[1:]
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
