import numpy as np


def upwinding(
    v: np.ndarray,
    left_value: np.ndarray,
    right_value: np.ndarray,
) -> float:
    """
    args:
        velocity    advection velocity defined at interface     (m, n, p, ...)
        left_value  value to the left of interface              (m, n, p, ...)
        left_value  value to the left of interface              (m, n, p, ...)
    returns:
        solution to riemann problem                             (m, n, p, ...)
    """
    left_flux, right_flux = v * left_value, v * right_value
    return ((right_flux + left_flux) - np.abs(v) * (right_value - left_value)) / 2.0
