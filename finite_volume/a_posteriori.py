import numpy as np
from typing import Callable, Tuple
from finite_volume.utils import chopchop


def minmod(du_left: np.ndarray, du_right: np.ndarray) -> np.ndarray:
    """
    minmod slope limiter, returns limited slopes of same shape as input
    """
    ratio = du_right / np.where(
        du_left > 0,
        np.where(du_left > 1e-16, du_left, 1e-16),
        np.where(du_left < -1e-16, du_left, -1e-16),
    )
    ratio = np.where(ratio < 1, ratio, 1)
    return np.where(ratio > 0, ratio, 0) * du_left


def moncen(du_left: np.ndarray, du_right: np.ndarray) -> np.ndarray:
    """
    moncen slope limiter, returns limited slopes of same shape as input
    """
    du_central = 0.5 * (du_left + du_right)
    slope = np.minimum(np.abs(2 * du_left), np.abs(2 * du_right))
    slope = np.sign(du_central) * np.minimum(slope, np.abs(du_central))
    return np.where(du_left * du_right >= 0, slope, 0)


def MUSCL(
    u: np.ndarray, axis: int, slope_limiter: Callable = minmod
) -> Tuple[np.ndarray, np.ndarray]:
    """
    second order fallback scheme for interpolating face values along axis
    args:
        u               (,m,)
        axis
        slope_limiter    minmod or moncen
    returns:
        slopes          (,m-2,)
    """
    du_left = chopchop(u, chop_size=(1, 1), axis=axis) - chopchop(
        u, chop_size=(0, 2), axis=axis
    )
    du_right = chopchop(u, chop_size=(2, 0), axis=axis) - chopchop(
        u, chop_size=(1, 1), axis=axis
    )
    du_limited = slope_limiter(du_left, du_right)
    return du_limited
