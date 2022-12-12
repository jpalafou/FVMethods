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
        self.ghost_width = (
            max(right_interface_scheme_original.coeffs.keys()) + 1
        )  # assumes symmetric scheme

    def update_boundary(self, x_extended: np.ndarray):
        gw = self.ghost_width
        # periodic boundary
        left_index = -2 * gw
        x_extended[:gw] = x_extended[left_index:-gw]
        right_index = 2 * gw
        x_extended[-gw:] = x_extended[gw:right_index]

    def xdot(self, x_i: np.ndarray, t_i: float) -> np.ndarray:
        gw = self.ghost_width
        # apply boundary conditions
        x_extended = np.concatenate((np.zeros(gw), x_i, np.zeros(gw)))
        self.update_boundary(x_extended)
        # reconstruct polynomial at cell faces
        n = len(x_extended)
        a = []
        for i in range(-(gw - 1), (gw - 1) + 1):
            left_index = gw - 1 + i
            right_index = n - gw + 1 + i
            a.append(x_extended[left_index:right_index])
        A = np.array(a).T
        right_interface_values = (
            A @ self.right_interface_scheme / sum(self.right_interface_scheme)
        )
        left_interface_values = (
            A @ self.left_interface_scheme / sum(self.left_interface_scheme)
        )
        # find the difference in each cell using a rudimentary reimann ssolver
        x_difference = np.zeros(len(self.x0))
        assert len(right_interface_values) - 2 == len(self.x0)
        for i in range(1, A.shape[0] - 1):
            # reimann solver
            if self.a > 0:
                x_difference[i - 1] = (
                    right_interface_values[i] - right_interface_values[i - 1]
                )
            elif self.a < 0:
                x_difference[i - 1] = (
                    left_interface_values[i + 1] - left_interface_values[i]
                )
        return -(self.a / self.h) * x_difference
