import numpy as np
from finite_volume.utils import chopchop


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
    ddu = (
        chopchop(du, chop_size=(2, 0), axis=axis)
        - chopchop(du, chop_size=(0, 2), axis=axis)
    ) / (2 * h)
    dv = 0.5 * h * ddu

    v_left = chopchop(du, chop_size=(0, 2), axis=axis) - chopchop(
        du, chop_size=(1, 1), axis=axis
    )
    alpha_left = -np.where(
        dv < 0, np.where(v_left > 0, v_left, 0), np.where(v_left < 0, v_left, 0)
    ) / np.where(np.abs(dv) < 1e-16, 1e-16 * np.where(dv >= 0, 1.0, -1.0), dv)
    alpha_left = np.where(np.abs(dv) <= 0, 1, alpha_left)
    alpha_left = np.where(alpha_left < 1, alpha_left, 1)
    v_right = chopchop(du, chop_size=(2, 0), axis=axis) - chopchop(
        du, chop_size=(1, 1), axis=axis
    )
    alpha_right = np.where(
        dv > 0, np.where(v_right > 0, v_right, 0), np.where(v_right < 0, v_right, 0)
    ) / np.where(np.abs(dv) < 1e-16, 1e-16 * np.where(dv >= 0, 1.0, -1.0), dv)
    alpha_right = np.where(np.abs(dv) <= 0, 1, alpha_right)
    alpha_right = np.where(alpha_right < 1, alpha_right, 1)
    alpha = np.where(alpha_left < alpha_right, alpha_left, alpha_right)

    # find minimum of alpha and neighbors
    alpha_min = np.amin(
        np.array(
            [
                chopchop(alpha, chop_size=(2, 0), axis=axis),
                chopchop(alpha, chop_size=(1, 1), axis=axis),
                chopchop(alpha, chop_size=(0, 2), axis=axis),
            ]
        ),
        axis=0,
    )
    return alpha_min


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


def compute_alpha_2d(u, zeros: bool = None):
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
    alpha = np.where(alpha_x < alpha_y, alpha_x, alpha_y)
    return alpha
