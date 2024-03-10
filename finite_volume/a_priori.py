"""
defines functions for computing the a priori slope limiting parameter, theta
"""

import numpy as np
from finite_volume.utils import f_of_3_neighbors, f_of_5_neighbors, np_floor


def mpp_limiter(
    u: np.ndarray, points: np.ndarray, zeros: bool = False, ones: bool = False
):
    """
    args:
        u       cell volume averages with padding       (m + 2,) or (m + 2, n + 2)
        points  u interpolated at quadrature points     (p, m) or (p, q, m, n)
        zeros   whether to return all 0s
        ones   whether to return all 1s
    returns:
        theta   slope limiting factor                   (m,) or (m, n)
        M_ij    cellwise maximum interpolation          (m,) or (m, n)
        m_ij    cellwise minimum interpolation          (m,) or (m, n)
    """

    # setup
    if u.ndim == 1:
        f_of_neighbors = f_of_3_neighbors
        u_inner = u[1:-1]
    elif u.ndim == 2:
        f_of_neighbors = f_of_5_neighbors
        u_inner = u[1:-1, 1:-1]

    if zeros:
        return np.zeros_like(u_inner), np.nan, np.nan
    if ones:
        return np.ones_like(u_inner), np.nan, np.nan

    # max and min of immediate neighbors
    M = f_of_neighbors(u, f=np.maximum)
    m = f_of_neighbors(u, f=np.minimum)

    # max and min of u evaluated at quadrature points
    M_ij = np.amax(points, axis=tuple(range(points.ndim - u.ndim)))
    m_ij = np.amin(points, axis=tuple(range(points.ndim - u.ndim)))

    # evaluate slope limiter
    theta = np.ones_like(u_inner)
    M_arg = np.abs(M - u_inner) / np_floor(np.abs(M_ij - u_inner), 1e-16)
    m_arg = np.abs(m - u_inner) / np_floor(np.abs(m_ij - u_inner), 1e-16)
    theta = np.minimum(np.minimum(M_arg, m_arg), theta)
    return theta, M_ij, m_ij
