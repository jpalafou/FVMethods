import numpy as np
from util.integrate import Integrator
from util.fvscheme import PolynomialReconstruction


class AdvectionSolver(Integrator):
    """
    uses polynomial reconstruction of arbitrary order
    no limiter
    """

    def __init__(
        self,
        x0: np.ndarray,
        t: np.ndarray,
        h: float,
        a: float,
        order: int = 1,
        bc_type: str = "periodic",
    ):
        super().__init__(x0, t)
        self.h = h  # mesh size
        self.a = a  # velocity field
        # devise a scheme for reconstructed values at cell interfaces
        right_interface_scheme_original = (
            PolynomialReconstruction.construct_from_order(order, "right")
        )
        left_interface_scheme_original = (
            PolynomialReconstruction.construct_from_order(order, "left")
        )
        self.right_interface_scheme = right_interface_scheme_original.nparray()
        self.left_interface_scheme = left_interface_scheme_original.nparray()
        # number of kernel cells from center referenced by a scheme
        # symmetry is assumed
        self._k = max(right_interface_scheme_original.coeffs.keys())
        # number of ghost cells on either side of the extended state vector
        self._gw = self._k + 1
        self.bc_type = bc_type  # type of boudnary condition

    def apply_bc(self, x_extended: np.ndarray):
        if self.bc_type == "periodic":
            gw = self._gw
            negative_gw = -gw
            left_index = -2 * gw
            x_extended[:gw] = x_extended[left_index:negative_gw]
            right_index = 2 * gw
            x_extended[negative_gw:] = x_extended[gw:right_index]

    def xdot(self, x: np.ndarray, t_i: float) -> np.ndarray:
        x_extended = np.concatenate(
            (np.zeros(self._gw), x, np.zeros(self._gw))
        )
        self.apply_bc(x_extended)
        a = []
        n = len(x)
        for i in range(2 * self._k + 1):
            right_ind = i + n + 2
            a.append(x_extended[i:right_ind])
        A = np.array(a).T
        x_interface_right = (
            A @ self.right_interface_scheme / sum(self.right_interface_scheme)
        )
        x_interface_left = (
            A @ self.left_interface_scheme / sum(self.left_interface_scheme)
        )
        Delta_x = np.zeros(n)
        for i in range(n):
            if self.a > 0:
                Delta_x[i] = x_interface_right[i + 1] - x_interface_right[i]
            elif self.a < 0:
                Delta_x[i] = x_interface_left[i + 2] - x_interface_left[i + 1]
        return -(self.a / self.h) * Delta_x


class AdvectionSolver_minmod(Integrator):
    """
    uses minmod limter
    """

    def __init__(
        self,
        x0: np.ndarray,
        t: np.ndarray,
        h: float,
        a: float,
        bc_type: str = "periodic",
    ):
        super().__init__(x0, t)
        self.h = h  # mesh size
        self.a = a  # velocity field
        # devise a scheme for reconstructed values at cell interfaces
        # number of ghost cells on either side of the extended state vector
        self.bc_type = bc_type  # type of boudnary condition

    def apply_bc(self, x_extended: np.ndarray):
        if self.bc_type == "periodic":
            x_extended[:2] = x_extended[-4:-2]
            x_extended[-2:] = x_extended[2:4]

    def xdot(self, x: np.ndarray, t_i: float) -> np.ndarray:
        x_extended = np.concatenate((np.zeros(2), x, np.zeros(2)))
        self.apply_bc(x_extended)
        Delta_x_left = x_extended[1:-1] - x_extended[:-2]
        Delta_x_right = x_extended[2:] - x_extended[1:-1]
        n = len(x)
        Delta_x_i = np.zeros(n + 2)  # \Delta x in the ith cell
        for i in range(n + 2):
            # if not Delta_x_left[i] * Delta_x_right[i] < 0:
            #     Delta_x_i[i] = min(Delta_x_left[i], Delta_x_right[i])

            # Delta_x_i[i] = min(Delta_x_left[i], Delta_x_right[i])

            if not Delta_x_left[i] * Delta_x_right[i] < 0:
                if abs(Delta_x_left[i]) < abs(Delta_x_right[i]):
                    Delta_x_i[i] = Delta_x_left[i]
                else:
                    Delta_x_i[i] = Delta_x_right[i]

            # if abs(Delta_x_left[i]) < abs(Delta_x_right[i]):
            #     Delta_x_i[i] = Delta_x_left[i]
            # else:
            #     Delta_x_i[i] = Delta_x_right[i]
        x_interface_right = x_extended[1:-1] + Delta_x_i / 2
        x_interface_left = x_extended[1:-1] - Delta_x_i / 2
        Delta_x = np.zeros(n)
        for i in range(n):
            if self.a > 0:
                Delta_x[i] = x_interface_right[i + 1] - x_interface_right[i]
            elif self.a < 0:
                Delta_x[i] = x_interface_left[i + 2] - x_interface_left[i + 1]
        return -(self.a / self.h) * Delta_x
