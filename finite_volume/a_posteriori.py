import numpy as np
from typing import Tuple
from finite_volume.sed import compute_alpha_1d, compute_alpha_2d
from finite_volume.utils import f_of_3_neighbors, f_of_4_neighbors


def find_trouble(
    u: np.ndarray,
    u_candidate: np.ndarray,
    NAD: float,
    PAD: Tuple[float, float],
    SED: bool = False,
    ones: bool = None,
):
    """
    args:
        u               current cell averages with padding  (m + 2,) or (m + 2, n + 2)
        u_candidate     candidate values with padding       (m + 6,) or (m + 6, n + 6)
        NAD             numerical admissibility detection
        PAD             physical admissibility detection
                        (lower, upper)
        SED             whether to use SED
        ones            return all trouble
    overwrites:
        trouble         boolean array                       (m,) or (m, n)
    """
    # setup
    if u.ndim == 1:
        f_of_neighbors = f_of_3_neighbors
        compute_alpha = compute_alpha_1d
        u_candidate_inner = u_candidate[3:-3]
    elif u.ndim == 2:
        f_of_neighbors = f_of_4_neighbors
        compute_alpha = compute_alpha_2d
        u_candidate_inner = u_candidate[3:-3, 3:-3]

    if ones:
        return np.ones_like(u_candidate_inner, dtype=np.int64)

    # max and min of immediate neighbors
    M = f_of_neighbors(u, f=np.maximum)
    m = f_of_neighbors(u, f=np.minimum)

    # NAD
    u_range = np.max(u) - np.min(u)
    tolerance = NAD * u_range
    upper_differences, lower_differences = u_candidate_inner - M, u_candidate_inner - m
    NAD_trouble = np.where(lower_differences < -tolerance, 1, 0)
    NAD_trouble = np.where(upper_differences > tolerance, 1, NAD_trouble)

    # SED
    alpha = compute_alpha(u_candidate, zeros=not SED)

    # PAD then SED then NAD
    PAD_trouble = np.logical_or(
        u_candidate_inner < min(PAD), u_candidate_inner > max(PAD)
    )
    not_smooth_extrema = alpha < 1
    trouble = np.where(PAD_trouble, 1, np.where(not_smooth_extrema, NAD_trouble, 0))

    return trouble


def minmod(du_left: np.ndarray, du_right: np.ndarray) -> np.ndarray:
    """
    args:
        du_left     left difference
        du_right    right difference
    returns:
        minmod limited difference
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
    args:
        du_left     left difference
        du_right    right difference
    returns:
        moncen limited difference
    """
    du_central = 0.5 * (du_left + du_right)
    slope = np.minimum(np.abs(2 * du_left), np.abs(2 * du_right))
    slope = np.sign(du_central) * np.minimum(slope, np.abs(du_central))
    return np.where(du_left * du_right >= 0, slope, 0)


def PAD_fallback(
    x: np.ndarray, fallback: np.ndarray, PAD: Tuple[float, float]
) -> np.ndarray:
    """
    args:
        x           any shape
        fallback    same shape as x
        PAD         (lower, upper)
    return:
        out         x where elements not bounded by PAD are replaced with fallback
    """
    PAD_violation = np.logical_or(x < min(PAD), x > max(PAD))
    return np.where(PAD_violation, fallback, x)


def compute_MUSCL_interpolations_1d(
    u: np.ndarray,
    slope_limiter: callable,
    fallback_to_1st_order: bool = False,
    PAD: Tuple[float, float] = None,
    hancock: bool = False,
    dt: float = None,
    h: float = None,
    v_cell_centers: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    args:
        u                       cell volme averages with padding    (m + 2,)
        slope_limiter           f(du_left, du_right)
        fallback_to_1st_order   whether to fall back to 1st order
                                in the presence of a PAD violation
        PAD                     (lower, upper)
        hancock                 whether to include redictor step
        dt                      time step size
        h                       mesh size
        v_cell_centers          velocity at cell centers            (m,)
    overwrites:
        left_face               left interpolated face value        (m,)
        right_face              right interpolated face value       (m,)
    """
    # compute second order slopes
    cell_centers = np.copy(u[1:-1])
    du_left = cell_centers - u[:-2]
    du_right = u[2:] - cell_centers
    du = slope_limiter(du_left, du_right)

    # apply predictor corrector scheme or dont
    if hancock:
        cell_centers -= 0.5 * v_cell_centers * (dt / h) * du

    # interpolate cell faces
    right_face = cell_centers + 0.5 * du
    left_face = cell_centers - 0.5 * du

    # fall back to first order if there are positivity violations
    if fallback_to_1st_order:
        right_face = PAD_fallback(right_face, cell_centers, PAD)
        left_face = PAD_fallback(left_face, cell_centers, PAD)
    return left_face, right_face


def compute_MUSCL_interpolations_2d(
    u: np.ndarray,
    slope_limiter: callable,
    fallback_to_1st_order: bool = False,
    PAD: Tuple[float, float] = None,
    hancock: bool = False,
    dt: float = None,
    h: Tuple[float, float] = None,
    v_cell_centers: Tuple[np.ndarray, np.ndarray] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    args:
        u                       cell volme averages with padding    (m + 2, n + 2)
        slope_limiter           f(du_left, du_right)
        fallback_to_1st_order   whether to fall back to 1st order
                                in the presence of a PAD violation
        PAD                     (lower, upper)
        hancock                 whether to include redictor step
        dt                      time step size
        h                       mesh sizes
            h_x
            h_y
        v_cell_centers          velocity at cell centers
            v_x                                                     (m, n)
            v_y                                                     ...
    overwrites:
        x faces
            west_face           left interpolated face value        (m, n)
            east_face           right interpolated face value       ...
        y faces
            south_face          bottom interpolated face value      ...
            north_face          top interpolated face value         ...
    """
    # compute second order slopes
    cell_centers = np.copy(u[1:-1, 1:-1])
    du_x = slope_limiter(cell_centers - u[1:-1, :-2], u[1:-1, 2:] - cell_centers)
    du_y = slope_limiter(cell_centers - u[:-2, 1:-1], u[2:, 1:-1] - cell_centers)

    # print(cell_centers)
    # print(du_x)
    # print(du_y)

    if hancock:
        dudx = v_cell_centers[0] * (1 / h[0]) * du_x
        dudy = v_cell_centers[1] * (1 / h[1]) * du_y
        cell_centers -= 0.5 * dt * (dudx + dudy)

    # interpolate face values
    north_face = cell_centers + 0.5 * du_y
    south_face = cell_centers - 0.5 * du_y
    east_face = cell_centers + 0.5 * du_x
    west_face = cell_centers - 0.5 * du_x

    # fall back to first order if there are positivity violations
    if fallback_to_1st_order:
        north_face = PAD_fallback(north_face, cell_centers, PAD)
        south_face = PAD_fallback(south_face, cell_centers, PAD)
        east_face = PAD_fallback(east_face, cell_centers, PAD)
        west_face = PAD_fallback(west_face, cell_centers, PAD)

    return (west_face, east_face), (south_face, north_face)


def compute_PP2D_interpolations(
    u: np.ndarray,
    hancock: bool = False,
    dt: float = None,
    h: Tuple[float, float] = None,
    v_cell_centers: Tuple[np.ndarray, np.ndarray] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    args:
        u                       cell volme averages with padding    (m + 2, n + 2)
        hancock                 whether to include redictor step
        dt                      time step size
        h                       mesh sizes
            h_x
            h_y
        v_cell_centers          velocity at cell centers
            v_x                                                     (m, n)
            v_y                                                     ...
    overwrites:
        x faces
            west_face           left interpolated face value        (m, n)
            east_face           right interpolated face value       ...
        y faces
            south_face          bottom interpolated face value      ...
            north_face          top interpolated face value         ...
    """
    # second order slopes
    Sx = 0.5 * (u[1:-1, 2:] - u[1:-1, :-2])
    Sy = 0.5 * (u[2:, 1:-1] - u[:-2, 1:-1])

    # gather neighbors
    cell_centers = np.copy(u[1:-1, 1:-1])
    list_of_8_neighbor_differences = [
        u[:-2, 1:-1] - cell_centers,
        u[2:, 1:-1] - cell_centers,
        u[1:-1, :-2] - cell_centers,
        u[1:-1, 2:] - cell_centers,
        u[2:, 2:] - cell_centers,
        u[2:, :-2] - cell_centers,
        u[:-2, 2:] - cell_centers,
        u[:-2, :-2] - cell_centers,
    ]
    eps = 1e-20
    V_min = np.minimum(np.minimum.reduce(list_of_8_neighbor_differences), -eps)
    V_max = np.maximum(np.maximum.reduce(list_of_8_neighbor_differences), eps)

    # limit slopes
    V = 2 * np.minimum(np.abs(V_min), np.abs(V_max)) / (np.abs(Sx) + np.abs(Sy) + eps)
    Sx = np.minimum(V, 1.0) * Sx
    Sy = np.minimum(V, 1.0) * Sy

    # predictor corrector scheme
    if hancock:
        dudx = v_cell_centers[0] * (1 / h[0]) * Sx
        dudy = v_cell_centers[1] * (1 / h[1]) * Sy
        cell_centers -= 0.5 * dt * (dudx + dudy)

    # interpolate faces
    north_face = cell_centers + 0.5 * Sy
    south_face = cell_centers - 0.5 * Sy
    east_face = cell_centers + 0.5 * Sx
    west_face = cell_centers - 0.5 * Sx

    return (west_face, east_face), (south_face, north_face)


def broadcast_troubled_cells_to_faces_1d(trouble: np.ndarray) -> np.ndarray:
    """
    args:
        trouble             cellwise trouble boolean    (m,)
    returns:
        troubled_interface  facewise trouble boolean    (m + 1,)
    """
    troubled_interface = np.zeros(trouble.shape[0] + 1, dtype=int)
    troubled_interface[:-1] = trouble
    troubled_interface[1:] = np.where(trouble, 1, troubled_interface[1:])
    return troubled_interface


def broadcast_troubled_cells_to_faces_with_blending_1d(
    trouble: np.ndarray,
) -> np.ndarray:
    """
    args:
        trouble                     cellwise trouble boolean with padding   (m + 4,)
    returns:
        troubled_interface_mask     facewise trouble mask                   (m + 1,)
    """
    # initialize theta
    troubled_interface_mask = np.zeros(trouble.shape[0] - 3, dtype=float)
    theta = trouble.astype("float")

    # First neighbors
    theta[:-1] = np.maximum(0.75 * trouble[1:], theta[:-1])
    theta[1:] = np.maximum(0.75 * trouble[:-1], theta[1:])

    # Second neighbors
    theta[:-1] = np.maximum(0.25 * (theta[1:] > 0), theta[:-1])
    theta[1:] = np.maximum(0.25 * (theta[:-1] > 0), theta[1:])

    # flag affected faces with theta
    troubled_interface_mask[...] = np.maximum(theta[1:-2], theta[2:-1])
    return troubled_interface_mask


def broadcast_troubled_cells_to_faces_2d(
    trouble: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    args:
        trouble                 trouble cellwise trouble boolean    (m, n)
    returns:
        troubled_interface_x    facewise trouble boolean            (m, n + 1)
        troubled_interface_y    facewise trouble boolean            (m + 1, n)
    """
    # flag faces of troubled cells as troubled
    troubled_interface_x = np.zeros((trouble.shape[0], trouble.shape[1] + 1), dtype=int)
    troubled_interface_y = np.zeros((trouble.shape[0] + 1, trouble.shape[0]), dtype=int)
    troubled_interface_x[:, :-1] = trouble
    troubled_interface_x[:, 1:] = np.where(trouble, 1, troubled_interface_x[:, 1:])
    troubled_interface_y[:-1, :] = trouble
    troubled_interface_y[1:, :] = np.where(trouble, 1, troubled_interface_y[1:, :])
    return troubled_interface_x, troubled_interface_y


def broadcast_troubled_cells_to_faces_with_blending_2d(
    trouble: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    args:
        trouble                 trouble cellwise trouble boolean    (m + 4, n + 4)
                                with padding
    returns:
        interface_trouble_mask_x    facewise trouble mask           (m, n + 1)
        interface_trouble_mask_y    facewise trouble mask           (m + 1, n)
    """
    # initialize theta
    interface_trouble_mask_x = np.zeros(
        (trouble.shape[0] - 4, trouble.shape[1] - 3), dtype=float
    )
    interface_trouble_mask_y = np.zeros(
        (trouble.shape[0] - 3, trouble.shape[1] - 4), dtype=float
    )
    theta = trouble.astype("float")

    # First neighbors
    theta[:, :-1] = np.maximum(0.75 * trouble[:, 1:], theta[:, :-1])
    theta[:, 1:] = np.maximum(0.75 * trouble[:, :-1], theta[:, 1:])
    theta[:-1, :] = np.maximum(0.75 * trouble[1:, :], theta[:-1, :])
    theta[1:, :] = np.maximum(0.75 * trouble[:-1, :], theta[1:, :])
    # Second neighbors
    theta[:-1, :-1] = np.maximum(0.5 * trouble[1:, 1:], theta[:-1, :-1])
    theta[:-1, 1:] = np.maximum(0.5 * trouble[1:, :-1], theta[:-1, 1:])
    theta[1:, :-1] = np.maximum(0.5 * trouble[:-1, 1:], theta[1:, :-1])
    theta[1:, 1:] = np.maximum(0.5 * trouble[:-1, :-1], theta[1:, 1:])
    # Third neighbors
    theta[:, :-1] = np.maximum(0.25 * (theta[:, 1:] > 0), theta[:, :-1])
    theta[:, 1:] = np.maximum(0.25 * (theta[:, :-1] > 0), theta[:, 1:])
    theta[:-1, :] = np.maximum(0.25 * (theta[1:, :] > 0), theta[:-1, :])
    theta[1:, :] = np.maximum(0.25 * (theta[:-1, :] > 0), theta[1:, :])

    # flag affected faces with theta
    interface_trouble_mask_x[...] = np.maximum(theta[2:-2, 1:-2], theta[2:-2, 2:-1])
    interface_trouble_mask_y[...] = np.maximum(theta[1:-2, 2:-2], theta[2:-1, 2:-2])
    return interface_trouble_mask_x, interface_trouble_mask_y
