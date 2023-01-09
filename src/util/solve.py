import numpy as np
from util.integrate import Integrator
from util.fvscheme import PolynomialReconstruction


class AdvectionSolver(Integrator):
    def __init__(self, x0, t, h, a, order):
        super().__init__(x0, t)
        self.a = a  # velocity field
        self.h = h  # mesh size
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

    def periodic_boundary(self, x_extended: np.ndarray):
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
        self.periodic_boundary(x_extended)
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
