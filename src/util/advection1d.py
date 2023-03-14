# advection solution schems
import numpy as np
from util.initial_condition import initial_condition1d
from util.polynome import Polynome
from util.integrate import Integrator, rk4_Dt_adjust
from util.fvscheme import ConservativeInterpolation


# class definition
# OH YEAH MEGA ADVECTION CLASS TIME
class AdvectionSolver(Integrator):
    """
    uses polynomial reconstruction of arbitrary order
    no limiter
    """

    def __init__(
        self,
        u0: np.ndarray = None,
        u0_preset: str = "square",
        n: int = 32,
        x: tuple = (0, 1),
        T: float = 2,
        a: float = 1,
        courant: float = 0.5,
        order: int = 1,
        bc_type: str = "periodic",
        apriori: str = None,
        aposteriori: bool = False,
    ):
        self.order = order
        self.apriori = apriori
        self.aposteriori = aposteriori

        # spatial discretization
        self.n = n
        self.x_interface = np.linspace(x[0], x[1], num=n + 1)
        self.x = 0.5 * (
            self.x_interface[:-1] + self.x_interface[1:]
        )  # x at cell centers
        self.h = (x[1] - x[0]) / n

        # advection velocty
        if isinstance(a, int) or isinstance(a, float):  # constant advection
            self.a = a * np.ones(len(self.x_interface))
            self.a_max = a
        else:  # advection vector
            if len(a) != len(self.x_interface):
                raise BaseException(
                    f"Invalid advection velocity array with size {len(a)}"
                )
            self.a = a
            self.a_max = np.max(np.abs(a))

        # time discretization
        self.courant = courant
        Dt = courant * self.h / max(max(np.abs(self.a)), 1e-6)
        time_step_adjustment = 1
        if self.order > 4:
            time_step_adjustment = rk4_Dt_adjust(
                self.h, self.x_interface[-1] - self.x_interface[0], self.order
            )
            print(
                f"Decreasing timestep by a factor of"
                f" {time_step_adjustment} to maintain order {self.order}"
                " with rk4"
            )
            Dt = Dt * time_step_adjustment
        self.time_step_adjustment = time_step_adjustment
        n_time = int(np.ceil(T / Dt))
        self.Dt = T / n_time
        self.t = np.linspace(0, T, num=n_time)

        # initial condition
        if not u0:
            self.u0 = initial_condition1d(self.x, u0_preset)
        else:
            if len(u0) != n:
                raise BaseException(
                    f"Invalid initial condition with size {len(u0)}"
                )
            self.u0 = u0

        # initialize integrator
        super().__init__(self.u0, self.t)

        # stensil for reconstructed values at cell interfaces
        right_interface_stensil_original = (
            ConservativeInterpolation.construct_from_order(self.order, "right")
        )
        left_interface_stensil_original = (
            ConservativeInterpolation.construct_from_order(self.order, "left")
        )
        self.right_interface_stensil = (
            right_interface_stensil_original.nparray()
        )
        self.left_interface_stensil = left_interface_stensil_original.nparray()
        # number of kernel cells from center referenced by a symmetric scheme
        self._k = max(right_interface_stensil_original.coeffs.keys())
        # number of ghost cells on either side of the extended state vector
        self._gw = self._k + 1
        self._stensil_size = 2 * self._k + 1
        self.bc_type = bc_type  # type of boudnary condition
        self.list_of_interior_stensils = []

        # stensils to interpolate at Guass-Lobatto quadrature points
        if apriori and "mpp" in apriori:
            # devise schemes for reconstructed values at free abscissas
            # find q, the number of abscissas
            p = self.order - 1  # degree p
            # how many quadrature points do we need for this degrees
            q = int(np.ceil((p + 1) / 2)) + 1
            self.q = q
            # find the free (interior) abscissas
            # solve on [0, 1] to find an exact origin solution in the case
            # of an odd legendre polynomial, enforce symmetry
            free_abscissas = (
                Polynome.legendre(q - 1).prime().find_zeros([0, 1])
            )
            if len(free_abscissas) > 0:
                if free_abscissas[0] == 0:
                    free_abscissas = [
                        -free_abscissas[len(free_abscissas) - i - 1]
                        for i in range(len(free_abscissas) - 1)
                    ] + free_abscissas
                else:
                    free_abscissas = [
                        -free_abscissas[len(free_abscissas) - i - 1]
                        for i in range(len(free_abscissas))
                    ] + free_abscissas
            assert len(free_abscissas) == q - 2
            # find quadrature weights
            # https://mathworld.wolfram.com/LobattoQuadrature.html
            endpoint_weight = 2 / (q * (q - 1))
            weights = (
                [endpoint_weight]
                + [
                    2 / (q * (q - 1) * Polynome.legendre(q - 1).eval(x) ** 2)
                    for x in free_abscissas
                ]
                + [endpoint_weight]
            )
            # divide by 2 for coordinate transformation
            transformed_free_abscissas = np.array(free_abscissas) / 2
            if self.apriori == "mpp lite":
                transformed_free_abscissas = [
                    0
                ]  # only evaluate at cell center
            weights = np.array(weights) / 2
            # now we can find a stability condition
            C_max = weights[0]
            self.C_max = C_max
            # determine if stability is satisfied
            Dt_max = max(
                [self.t[i + 1] - self.t[i] for i in range(len(self.t) - 1)]
            )  # maximum time step
            if self.a_max * Dt_max / self.h >= C_max:
                print(
                    "WARNING: Maximum principle preserving not satisfied.\n"
                    f"Try a timestep less than {C_max * self.h / self.a_max}"
                    f" for a Courant condition of {C_max}."
                )
            # key takeway from the Gauss-Lobatto quadrature are the schemes
            # to evaluate at the interior quadrature points
            list_of_interior_stensils = []
            for x in transformed_free_abscissas:
                list_of_interior_stensils.append(
                    ConservativeInterpolation.construct_from_order(
                        order, x
                    ).nparray()
                )
            # if the scheme is missing elements place a zero on either end
            list_of_interior_stensils_same_length = []
            for stensil in list_of_interior_stensils:
                if len(stensil) < self._stensil_size:
                    stensil = np.concatenate(
                        (np.zeros(1), stensil, np.zeros(1))
                    )
                    assert len(stensil) == self._stensil_size
                list_of_interior_stensils_same_length.append(stensil)
            self.list_of_interior_stensils = (
                list_of_interior_stensils_same_length
            )

    def apply_bc(
        self, without_ghost_cells: np.ndarray, num_ghost_cells: int
    ) -> np.ndarray:
        gw = num_ghost_cells
        negative_gw = -gw
        with_ghost_cells = np.zeros(len(without_ghost_cells) + 2 * gw)
        with_ghost_cells[gw:negative_gw] = without_ghost_cells
        if self.bc_type == "periodic":
            left_index = -2 * gw
            with_ghost_cells[:gw] = with_ghost_cells[left_index:negative_gw]
            right_index = 2 * gw
            with_ghost_cells[negative_gw:] = with_ghost_cells[gw:right_index]
        return with_ghost_cells

    def reimann(
        self,
        velocities_at_boundaries: float,
        left_of_boundary_values: float,
        right_of_boundary_values: float,
    ) -> float:
        fluxes = np.zeros(len(velocities_at_boundaries))
        fluxes = np.where(
            velocities_at_boundaries > 0, left_of_boundary_values, fluxes
        )
        fluxes = np.where(
            velocities_at_boundaries < 0, right_of_boundary_values, fluxes
        )
        return fluxes * velocities_at_boundaries

    def udot(self, u: np.ndarray, t_i: float) -> np.ndarray:
        ubar_extended = self.apply_bc(u, self._gw)
        list_of_windows = []
        n = len(u)
        nwg = len(ubar_extended)
        # construct an array of staggered state vectors such that a
        # matrix operation with a stensil multiplies weights by their
        # referenced cells
        # right and left interface interpolations
        for i in range(self._stensil_size):
            right_ind = i + nwg - (self._stensil_size - 1)
            list_of_windows.append(ubar_extended[i:right_ind])
        array_of_windows = np.array(list_of_windows).T
        # interpolate at right and left cell interfaces
        u_interface_right = (
            array_of_windows
            @ self.right_interface_stensil
            / sum(self.right_interface_stensil)
        )
        u_interface_left = (
            array_of_windows
            @ self.left_interface_stensil
            / sum(self.left_interface_stensil)
        )
        # interpolate using interior stensils
        list_of_interior_evaluations = []
        for stensil in self.list_of_interior_stensils:
            list_of_interior_evaluations.append(
                array_of_windows @ stensil / sum(stensil)
            )
        array_of_interior_evaluations = np.array(
            list_of_interior_evaluations
        ).reshape(len(self.list_of_interior_stensils), len(u_interface_right))
        evaluations = np.concatenate(
            (
                u_interface_left.reshape(1, n + 2),
                array_of_interior_evaluations.reshape(-1, n + 2),
                u_interface_right.reshape(1, n + 2),
            )
        )
        # slope limiter
        theta_i = np.ones(n + 2)
        ubar_1gw = self.apply_bc(u, 1)
        if self.apriori and "mpp" in self.apriori:
            ubar_2gw = self.apply_bc(u, 2)
            M = np.maximum(
                ubar_2gw[:-2], np.maximum(ubar_2gw[1:-1], ubar_2gw[2:])
            )
            m = np.minimum(
                ubar_2gw[:-2], np.minimum(ubar_2gw[1:-1], ubar_2gw[2:])
            )
            M_i = np.amax(evaluations, axis=0)
            m_i = np.amin(evaluations, axis=0)
            theta_arg_M = np.abs(M - ubar_1gw) / np.abs(M_i - ubar_1gw)
            theta_arg_m = np.abs(m - ubar_1gw) / np.abs(m_i - ubar_1gw)
            theta_i = np.where(theta_arg_M < theta_i, theta_arg_M, theta_i)
            theta_i = np.where(theta_arg_m < theta_i, theta_arg_m, theta_i)
        # slope limited interpolation at cell faces
        u_interface_right_tilda = (
            theta_i * (u_interface_right - ubar_1gw) + ubar_1gw
        )
        u_interface_left_tilda = (
            theta_i * (u_interface_left - ubar_1gw) + ubar_1gw
        )
        # right fluxes - left fluxes
        a_Delta_u = self.reimann(
            self.a[1:],
            u_interface_right_tilda[1:-1],
            u_interface_left_tilda[2:],
        ) - self.reimann(
            self.a[:-1],
            u_interface_right_tilda[:-2],
            u_interface_left_tilda[1:-1],
        )
        return -a_Delta_u / self.h