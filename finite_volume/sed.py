"""
defines functions for determining whether a cell is a "smooth extrema", where slope
limiting should be turned off
"""

import numpy as np
from finite_volume.utils import chopchop, avoid_0


def detect_smooth_extrema(u: np.ndarray, h: float = 1.0, axis: int = 0):
    """
    args:
        u   array of any shape, (..., m + 6, ...)
        h   mesh size in the axis direction
        axis
    returns:
        array of whether or not a cell is smooth (..., m, ...)
    """
    du = (
        chopchop(u, chop_size=(2, 0), axis=axis)
        - chopchop(u, chop_size=(0, 2), axis=axis)
    ) / (2 * h)
    S_left = (
        chopchop(du, chop_size=(1, 1), axis=axis)
        - chopchop(du, chop_size=(0, 2), axis=axis)
    ) / h
    S_right = (
        chopchop(du, chop_size=(2, 0), axis=axis)
        - chopchop(du, chop_size=(1, 1), axis=axis)
    ) / h
    S_center = avoid_0(0.5 * (S_left + S_right), 1e-10)

    def alpha_direction(S_direction):
        alpha_direction_min_min = np.minimum(
            1.0, np.minimum(2 * S_direction, 0.0) / S_center
        )
        alpha_direction_min_max = np.minimum(
            1.0, np.maximum(2 * S_direction, 0.0) / S_center
        )
        alpha_direction = np.where(
            S_center < 0.0,
            alpha_direction_min_min,
            np.where(S_center > 0.0, alpha_direction_min_max, 1.0),
        )
        return alpha_direction

    alpha_left = alpha_direction(S_left)
    alpha_right = alpha_direction(S_right)
    alpha = np.minimum(alpha_left, alpha_right)

    alpha = np.minimum(
        np.minimum(
            chopchop(alpha, chop_size=(0, 2), axis=axis),
            chopchop(alpha, chop_size=(1, 1), axis=axis),
        ),
        chopchop(alpha, chop_size=(2, 0), axis=axis),
    )

    return alpha


def compute_alpha_1d(u: np.ndarray, zeros: bool = None) -> np.ndarray:
    """
    args:
        u       cell volume averages (m + 6,)
        h       mesh size
        zeros   whether to return all 0s
    returns
        alpha   np array (m,)
    """
    if zeros:
        return np.zeros_like(u[3:-3])
    return detect_smooth_extrema(u, axis=0)


def compute_alpha_2d(u, zeros: bool = False):
    """
    args:
        u       cell volume averages (m + 6, n + 6)
        hx      mesh size in x (axis 1)
        hy      mesh size in y (axis 0)
        zeros   whether to return all 0s
    returns
        alpha   np array (m, n)
    """
    if zeros:
        return np.zeros_like(u[3:-3, 3:-3])
    alpha_x = detect_smooth_extrema(u, axis=1)[3:-3, :]
    alpha_y = detect_smooth_extrema(u, axis=0)[:, 3:-3]
    alpha = np.minimum(alpha_x, alpha_y)
    return alpha
