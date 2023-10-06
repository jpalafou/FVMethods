import numpy as np
from finite_volume.utils import chopchop


def detect_smooth_extrema(u: np.ndarray, h: float, axis: int):
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
    ) / np.maximum(dv, 1e-16 * np.ones_like(dv))
    alpha_left = np.where(np.abs(dv) <= 0, 1, alpha_left)
    alpha_left = np.where(alpha_left < 1, alpha_left, 1)

    v_right = chopchop(du, chop_size=(2, 0), axis=axis) - chopchop(
        du, chop_size=(1, 1), axis=axis
    )
    alpha_right = np.where(
        dv > 0, np.where(v_right > 0, v_right, 0), np.where(v_right < 0, v_right, 0)
    ) / np.maximum(dv, 1e-16 * np.ones_like(dv))
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


def detect_no_smooth_extrema(u: np.ndarray, axis: int, **kwargs):
    return chopchop(np.zeros_like(u), chop_size=(3, 3), axis=axis)


def minmod(du_left, du_right):
    """
    minmod slope limiter, returns limited slopes of same shape as input
    """
    ratio = du_right / np.maximum(du_left, 1e-16)
    ratio = np.where(ratio < 1, ratio, 1)
    return np.where(ratio > 0, ratio, 0) * du_left


def moncen(du_left, du_right):
    """
    moncen slope limiter, returns limited slopes of same shape as input
    """
    du_central = 0.5 * (du_left + du_right)
    slope = np.minimum(np.abs(2 * du_left), np.abs(2 * du_right))
    slope = np.sign(du_central) * np.minimum(slope, np.abs(du_central))
    return np.where(du_left * du_right >= 0, slope, 0)


def compute_fallback_faces(u, axis, compute_slopes=moncen):
    """
    second order fallback scheme for interpolating face values along axis
    args:
        u   (..., m + 2, ...)
        axis
        compute_slopes  function set to moncen for now
    returns:
        out (..., m, ...)
    """
    du = chopchop(u, chop_size=(1, 0), axis=axis) - chopchop(
        u, chop_size=(0, 1), axis=axis
    )
    du_left = chopchop(du, chop_size=(0, 1), axis=axis)
    du_right = chopchop(du, chop_size=(1, 0), axis=axis)
    limited_slopes = 0.5 * compute_slopes(du_left, du_right)
    left_face = chopchop(u, chop_size=(1, 1), axis=axis) - limited_slopes
    right_face = chopchop(u, chop_size=(1, 1), axis=axis) + limited_slopes
    return left_face, right_face
