# advection solution schems
import numpy as np
from util.initial_condition import initial_condition1d
from util.polynome import Polynome
from util.integrate import Integrator, rk4_Dt_adjust
from util.fvscheme import ConservativeInterpolation


# class definition
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
    ):
        self.a = a
        self.order = order

        # spatial discretization
        self.n = n
        self.x_interface = np.linspace(x[0], x[1], num=n + 1)
        self.x = 0.5 * (
            self.x_interface[:-1] + self.x_interface[1:]
        )  # x at cell centers
        self.h = (x[1] - x[0]) / n

        # time discretization
        self.courant = courant
        if a:  # nonzero advection velocity
            Dt = courant * self.h / np.abs(a)
        else:
            Dt = courant * self.h
        time_step_adjustment = 1
        if order > 4:
            time_step_adjustment = rk4_Dt_adjust(self.h, x[1] - x[0], order)
            print(
                f"Decreasing timestep by a factor of"
                f" {time_step_adjustment} to maintain order {order}"
                " with rk4"
            )
            Dt = Dt * time_step_adjustment
        n_time = int(np.ceil(T / Dt))
        self.Dt = T / n_time
        self.time_step_adjustment = time_step_adjustment
        self.t = np.linspace(0, T, num=n_time)

        # initial condition
        if not u0:
            self.u0 = initial_condition1d(self.x, u0_preset)
        else:
            if len(u0) != n:
                raise BaseException(f"Invalid initial condition for size {n}")
            self.u0 = u0

        # initialize integrator
        super().__init__(self.u0, self.t)

        # stensil for reconstructed values at cell interfaces
        right_interface_stensil_original = (
            ConservativeInterpolation.construct_from_order(order, "right")
        )
        left_interface_stensil_original = (
            ConservativeInterpolation.construct_from_order(order, "left")
        )
        self.right_interface_stensil = (
            right_interface_stensil_original.nparray()
        )
        self.left_interface_stensil = left_interface_stensil_original.nparray()
        # number of kernel cells from center referenced by a symmetric scheme
        self._k = max(right_interface_stensil_original.coeffs.keys())
        # number of ghost cells on either side of the extended state vector
        self._gw = self._k + 1
        self.bc_type = bc_type  # type of boudnary condition

    def apply_bc(self, ubar_extended: np.ndarray):
        if self.bc_type == "periodic":
            gw = self._gw
            negative_gw = -gw
            left_index = -2 * gw
            ubar_extended[:gw] = ubar_extended[left_index:negative_gw]
            right_index = 2 * gw
            ubar_extended[negative_gw:] = ubar_extended[gw:right_index]

    def reimann(
        self,
        value_to_the_left_of_boundary: float,
        value_to_the_right_of_boundary: float,
    ) -> float:
        if self.a > 0:
            return value_to_the_left_of_boundary
        elif self.a < 0:
            return value_to_the_right_of_boundary
        else:
            return 0

    def udot(self, u: np.ndarray, t_i: float) -> np.ndarray:
        ubar_extended = np.concatenate(
            (np.zeros(self._gw), u, np.zeros(self._gw))
        )
        self.apply_bc(ubar_extended)
        a = []
        n = len(u)
        # construct an array of staggered state vectors such that the
        # a matrix operation with a stensil multiplies weights by their
        # referenced cells
        for i in range(2 * self._k + 1):
            right_ind = i + n + 2
            a.append(ubar_extended[i:right_ind])
        A = np.array(a).T
        u_interface_right = (
            A
            @ self.right_interface_stensil
            / sum(self.right_interface_stensil)
        )
        u_interface_left = (
            A @ self.left_interface_stensil / sum(self.left_interface_stensil)
        )
        Delta_u = np.zeros(n)
        for i in range(n):
            Delta_u[i] = self.reimann(
                u_interface_right[i + 1], u_interface_left[i + 2]
            ) - self.reimann(u_interface_right[i], u_interface_left[i + 1])
        return -(self.a / self.h) * Delta_u


class AdvectionSolver_minmod(AdvectionSolver):
    """
    uses minmod limter
    """

    def __init__(
        self,
        u0: np.ndarray,
        t: np.ndarray,
        h: float,
        a: float,
        bc_type: str = "periodic",
    ):
        Integrator.__init__(self, u0, t)
        self.h = h  # mesh size
        self.a = a  # velocity field
        # devise a scheme for reconstructed values at cell interfaces
        # number of ghost cells on either side of the extended state vector
        self._gw = 2
        self.bc_type = bc_type  # type of boudnary condition

    def udot(self, u: np.ndarray, t_i: float) -> np.ndarray:
        ubar_extended = np.concatenate((np.zeros(2), u, np.zeros(2)))
        self.apply_bc(ubar_extended)
        Delta_u_left = ubar_extended[1:-1] - ubar_extended[:-2]
        Delta_u_right = ubar_extended[2:] - ubar_extended[1:-1]
        n = len(u)
        Delta_u_i = np.zeros(n + 2)  # \Delta x in the ith cell
        for i in range(n + 2):
            if not Delta_u_left[i] * Delta_u_right[i] < 0:
                if abs(Delta_u_left[i]) < abs(Delta_u_right[i]):
                    Delta_u_i[i] = Delta_u_left[i]
                else:
                    Delta_u_i[i] = Delta_u_right[i]
        u_interface_right = ubar_extended[1:-1] + Delta_u_i / 2
        u_interface_left = ubar_extended[1:-1] - Delta_u_i / 2
        Delta_u = np.zeros(n)
        for i in range(n):
            Delta_u[i] = self.reimann(
                u_interface_right[i + 1], u_interface_left[i + 2]
            ) - self.reimann(u_interface_right[i], u_interface_left[i + 1])
        return -(self.a / self.h) * Delta_u


class AdvectionSolver_nOrder_MPP(AdvectionSolver):
    """
    arbitrary order MPP slope limiter
    """

    def __init__(
        self,
        u0: np.ndarray,
        t: np.ndarray,
        h: float,
        a: float,
        order: int = 1,
        bc_type: str = "periodic",
    ):
        Integrator.__init__(self, u0, t)
        self.order = order  # order of scheme
        self.h = h  # mesh size
        self.a = a  # velocity field
        # devise a scheme for reconstructed values at cell interfaces
        right_interface_stensil_original = (
            ConservativeInterpolation.construct_from_order(order, "right")
        )
        left_interface_stensil_original = (
            ConservativeInterpolation.construct_from_order(order, "left")
        )
        self.right_interface_stensil = (
            right_interface_stensil_original.nparray()
        )
        self.left_interface_stensil = left_interface_stensil_original.nparray()
        # Guass-Lobatto quadrature ~ ~ ~
        # devise schemes for reconstructed values at free abscissas
        # find q, the number of abscissas
        p = order - 1  # degree p
        # how many quadrature points do we need for this degrees
        q = int(np.ceil((p + 1) / 2)) + 1
        self.q = q
        # find the free (interior) abscissas
        # solve on [0, 1] to find an exact origin solution in the case
        # of an odd legendre polynomial, enforce symmetry
        free_abscissas = Polynome.legendre(q - 1).prime().find_zeros([0, 1])
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
        weights = np.array(weights) / 2
        # now we can find a stability condition
        C_max = weights[0]
        self.C_max = C_max
        # determine if stability is satisfied
        Dt_max = max(
            [t[i + 1] - t[i] for i in range(len(t) - 1)]
        )  # maximum time step
        if a * Dt_max / h >= C_max:
            print(
                "WARNING: Maximum principle preserving not satisfied.\n"
                f"Try a timestep less than {C_max * h / a} for a "
                f"Courant condition of {C_max}."
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
        self.list_of_interior_stensils = list_of_interior_stensils
        # ~ ~ ~
        # number of kernel cells from center referenced by a scheme
        # symmetry is assumed
        self._k = max(right_interface_stensil_original.coeffs.keys())
        # number of ghost cells on either side of the extended state vector
        self._gw = self._k + 1 + 1
        self.bc_type = bc_type  # type of boudnary condition

    def udot(self, u: np.ndarray, t_i: float) -> np.ndarray:
        ubar_extended = np.concatenate(
            (np.zeros(self._gw), u, np.zeros(self._gw))
        )
        self.apply_bc(ubar_extended)
        a = []
        n = len(u)
        assert n + 4 + 2 * self._k == len(ubar_extended)
        for i in range(2 * self._k + 1):
            right_ind = n + 4 + i
            a.append(ubar_extended[i:right_ind])
        A = np.array(a).T
        u_interface_right = (
            A
            @ self.right_interface_stensil
            / sum(self.right_interface_stensil)
        )
        u_interface_left = (
            A @ self.left_interface_stensil / sum(self.left_interface_stensil)
        )
        list_of_interior_evaluations = []
        for stensil in self.list_of_interior_stensils:
            # if the scheme is of even order, place a zero on either end
            if len(stensil) < 2 * self._k + 1:
                stensil = np.concatenate((np.zeros(1), stensil, np.zeros(1)))
            list_of_interior_evaluations.append(A @ stensil / sum(stensil))
        array_of_interior_evaluations = np.array(list_of_interior_evaluations)
        if np.any(array_of_interior_evaluations):
            evaluations = np.concatenate(
                (
                    u_interface_left.reshape(1, n + 4),
                    array_of_interior_evaluations,
                    u_interface_right.reshape(1, n + 4),
                )
            )
        else:
            evaluations = np.concatenate(
                (
                    u_interface_left.reshape(1, n + 4),
                    u_interface_right.reshape(1, n + 4),
                )
            )
        M = np.zeros(n + 2)
        m = np.zeros(n + 2)
        M_i = np.zeros(n + 2)
        m_i = np.zeros(n + 2)
        i_shift = self._k
        for i in range(n + 2):
            M[i] = max(
                ubar_extended[i + i_shift],
                ubar_extended[i + i_shift + 1],
                ubar_extended[i + i_shift + 2],
            )
            m[i] = min(
                ubar_extended[i + i_shift],
                ubar_extended[i + i_shift + 1],
                ubar_extended[i + i_shift + 2],
            )
        assert max(evaluations.shape) == n + 4
        for i in range(n + 2):
            M_i[i] = max(evaluations[:, i + 1])
            m_i[i] = min(evaluations[:, i + 1])
        theta_i = np.zeros(n + 2)
        for i in range(n + 2):
            theta_i[i] = min(
                1,
                np.abs(M[i] - ubar_extended[self._k + 1 + i])
                / (np.abs(M_i[i] - ubar_extended[self._k + 1 + i]) + 1e-15),
                np.abs(m[i] - ubar_extended[self._k + 1 + i])
                / (np.abs(m_i[i] - ubar_extended[self._k + 1 + i]) + 1e-15),
            )
        left_index = self._k + 1
        right_index = -left_index
        u_interface_right_tilda = (
            theta_i
            * (u_interface_right[1:-1] - ubar_extended[left_index:right_index])
            + ubar_extended[left_index:right_index]
        )
        u_interface_left_tilda = (
            theta_i
            * (u_interface_left[1:-1] - ubar_extended[left_index:right_index])
            + ubar_extended[left_index:right_index]
        )
        assert len(u_interface_left_tilda) == n + 2
        assert len(u_interface_right_tilda) == n + 2
        Delta_u = np.zeros(n)
        for i in range(n):
            if self.a > 0:
                Delta_u[i] = self.reimann(
                    u_interface_right_tilda[i + 1],
                    u_interface_left_tilda[i + 2],
                ) - self.reimann(
                    u_interface_right_tilda[i], u_interface_left_tilda[i + 1]
                )
        return -(self.a / self.h) * Delta_u


class AdvectionSolver_nOrder_MPP_lite(AdvectionSolver_nOrder_MPP):
    """
    arbitrary order MPP slope limiter, the cell center is the only inner
    point evaluated
    """

    def __init__(
        self,
        u0: np.ndarray,
        t: np.ndarray,
        h: float,
        a: float,
        order: int = 1,
        bc_type: str = "periodic",
    ):
        Integrator.__init__(self, u0, t)
        self.order = order  # order of scheme
        self.h = h  # mesh size
        self.a = a  # velocity field
        # devise a scheme for reconstructed values at cell interfaces
        right_interface_stensil_original = (
            ConservativeInterpolation.construct_from_order(order, "right")
        )
        left_interface_stensil_original = (
            ConservativeInterpolation.construct_from_order(order, "left")
        )
        self.right_interface_stensil = (
            right_interface_stensil_original.nparray()
        )
        self.left_interface_stensil = left_interface_stensil_original.nparray()
        # Guass-Lobatto quadrature ~ ~ ~
        # devise schemes for reconstructed values at free abscissas
        # find q, the number of abscissas
        p = order - 1  # degree p
        # how many quadrature points do we need for this degrees
        q = int(np.ceil((p + 1) / 2)) + 1
        self.q = q
        # find the free (interior) abscissas
        transformed_free_abscissas = [0] if self.order > 2 else []

        # find quadrature weights
        # https://mathworld.wolfram.com/LobattoQuadrature.html
        endpoint_weight = 2 / (q * (q - 1))
        # transform coordinates
        endpoint_weight = endpoint_weight / 2
        # now we can find a stability condition
        C_max = endpoint_weight
        self.C_max = C_max
        # determine if stability is satisfied
        Dt_max = max(
            [t[i + 1] - t[i] for i in range(len(t) - 1)]
        )  # maximum time step
        if a * Dt_max / h >= C_max:
            print(
                "WARNING: Maximum principle preserving not satisfied.\n"
                f"Try a timestep less than {C_max * h / a} for a "
                f"Courant condition of {C_max}."
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
        self.list_of_interior_stensils = list_of_interior_stensils
        # ~ ~ ~
        # number of kernel cells from center referenced by a scheme
        # symmetry is assumed
        self._k = max(right_interface_stensil_original.coeffs.keys())
        # number of ghost cells on either side of the extended state vector
        self._gw = self._k + 1 + 1
        self.bc_type = bc_type  # type of boudnary condition
