# advection solution schems
import numpy as np
from util.integrate import Integrator
from util.fvscheme import ConservativeInterpolation


class AdvectionSolver2D(Integrator):
    """
    uses polynomial reconstruction of arbitrary order
    no limiter
    """

    def __init__(
        self,
        u0: np.ndarray,
        t: np.ndarray,
        h: float,
        a: float,
        b: float,
        order: int = 1,
        bc_type: str = "periodic",
    ):
        super().__init__(u0, t)
        self.h = h  # mesh size
        self.a = a  # horizontal velocity
        self.b = b  # vertical velocity
        # devise a scheme for reconstructed values at cell interfaces
        right_interface_stensil_original = (
            ConservativeInterpolation.construct_from_order(order, "right")
        )
        left_interface_stensil_original = (
            ConservativeInterpolation.construct_from_order(order, "left")
        )
        self.right_interface_stensil = (
            right_interface_stensil_original.nparray().reshape(-1, 1, 1)
        )
        self.left_interface_stensil = (
            left_interface_stensil_original.nparray().reshape(-1, 1, 1)
        )
        # number of kernel cells from center referenced by a scheme
        # symmetry is assumed
        self._k = max(right_interface_stensil_original.coeffs.keys())
        # number of ghost cells on either side of the extended state vector
        self._gw = self._k + 1
        self.bc_type = bc_type  # type of boudnary condition

    def apply_bc(self, ubarbar_extended: np.ndarray):
        if self.bc_type == "periodic":
            gw = self._gw
            negative_gw = -gw
            left_index = -2 * gw
            right_index = 2 * gw
            # left/right
            ubarbar_extended[:, :gw] = ubarbar_extended[
                :, left_index:negative_gw
            ]
            ubarbar_extended[:, negative_gw:] = ubarbar_extended[
                :, gw:right_index
            ]
            # up/down
            ubarbar_extended[:gw, :] = ubarbar_extended[
                left_index:negative_gw, :
            ]
            ubarbar_extended[negative_gw:, :] = ubarbar_extended[
                gw:right_index, :
            ]

    def reimann(
        self,
        a_at_boundary,
        value_to_the_left_of_boundary: float,
        value_to_the_right_of_boundary: float,
    ) -> float:
        if a_at_boundary > 0:
            return value_to_the_left_of_boundary
        elif a_at_boundary < 0:
            return value_to_the_right_of_boundary
        else:
            return 0

    def udot(self, u: np.ndarray, t_i: float) -> np.ndarray:
        # construct an extended array with ghost zones and apply bc
        n = len(u)
        gw = self._gw
        antigw = -self._gw
        ubarbar_extended = np.zeros((n + 2 * gw, n + 2 * gw))
        ubarbar_extended[gw:antigw, gw:antigw] = u
        self.apply_bc(ubarbar_extended)
        # construct an array of staggered state arrays such that a 3D
        # matrix operation with a stensil multiplies weights by their
        # referenced cells
        # north/south east/west
        ns = []
        ew = []
        stensil_length = 2 * self._k + 1
        for i in range(stensil_length):
            right_ind = i + n + 2
            ns.append(ubarbar_extended[i:right_ind, :])
            ew.append(ubarbar_extended[:, i:right_ind])
        NS = np.asarray(ns)
        EW = np.asarray(ew)
        ubar_north = sum(NS * self.left_interface_stensil) / sum(
            self.left_interface_stensil
        )
        ubar_south = sum(NS * self.right_interface_stensil) / sum(
            self.right_interface_stensil
        )
        ubar_east = sum(EW * self.right_interface_stensil) / sum(
            self.right_interface_stensil
        )
        ubar_west = sum(EW * self.left_interface_stensil) / sum(
            self.left_interface_stensil
        )

        Delta_u_NS = np.zeros((n, n))
        Delta_u_EW = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Delta_u_NS[i] = self.reimann(
                    self.b,
                    ubar_north[i + gw, j + gw],
                    ubar_south[i + gw - 1, j + gw],
                ) - self.reimann(
                    self.b,
                    ubar_north[i + gw + 1, j + gw],
                    ubar_south[i + gw, j + gw],
                )
                Delta_u_EW[i] = self.reimann(
                    self.a,
                    ubar_east[i + gw, j + gw],
                    ubar_west[i + gw, j + gw + 1],
                ) - self.reimann(
                    self.a,
                    ubar_east[i + gw, j + gw - 1],
                    ubar_west[i + gw, j + gw],
                )
        return -(self.b / self.h) * Delta_u_NS - (self.a / self.h) * Delta_u_EW
