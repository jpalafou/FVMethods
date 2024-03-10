"""
defines useful functions which I haven't bothered to categorize
"""

from itertools import product
from types import ModuleType
import numpy as np


def avoid_0(x: np.ndarray, eps: float, postive_at_0: bool = True) -> np.ndarray:
    """
    args:
        x:              array
        eps:            tolerance
        positive_at_0:  whether to use positive eps where x is 0
    returns:
        x with near-zero elements rounded to +eps or -eps depending on sign
    """
    if postive_at_0:
        negative_eps = np.logical_and(x > -eps, x < 0.0)
        positive_eps = np.logical_and(x >= 0.0, x < eps)
    else:
        negative_eps = np.logical_and(x > -eps, x <= 0.0)
        positive_eps = np.logical_and(x > 0.0, x < eps)
    return np.where(positive_eps, eps, np.where(negative_eps, -eps, x))


def chop(u: np.ndarray, chop_size: int, axis: int) -> np.ndarray:
    """
    symmetric chop of array edges about axis
    args:
        u:          np array of arbitrary dimension
        chop_size:  int
        axis:       int or list of axes
    returns:
        u:          symmetrically cut at the ends along axis by cut_length
    """
    index = np.array([slice(None)] * u.ndim)
    index[axis] = slice(chop_size, -chop_size or None)
    return u[tuple(index)]


def chopchop(u: np.ndarray, chop_size: tuple, axis: int):
    """
    asymmetric chop of array edges about axis
    args:
        u:          np array of arbitrary dimension
        chop_size:  tuple
        axis:       int or list of axes
    returns:
        u:          symmetrically cut at the ends along axis by cut_length
    """
    index = np.array([slice(None)] * u.ndim)
    index[axis] = slice(chop_size[0], -chop_size[1] or None)
    return u[tuple(index)]


def convolve2d(arr: np.ndarray, kernel: np.ndarray, xp: ModuleType = np) -> np.ndarray:
    """
    args:
        arr:        2D array with padding (m + p - 1, n + q - 1) for kernel
        kernel:     2D array (p, q)
        xp:         numpy or cupy
    returns:
        out:        arr convolved with kernel (m, n)
    """
    eaten = kernel.shape[0] - 1, kernel.shape[1] - 1
    windows = xp.empty((kernel.size, arr.shape[0] - eaten[0], arr.shape[1] - eaten[1]))
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            row_slice = slice(i, i + windows.shape[1])
            col_slice = slice(j, j + windows.shape[2])
            windows[j + kernel.shape[1] * i] = arr[row_slice, col_slice]
    out = np.sum(windows * kernel.reshape(kernel.size, 1, 1), axis=0)
    return out


def convolve_batch2d(
    arr: np.ndarray, kernel: np.ndarray, xp: ModuleType = np
) -> np.ndarray:
    """
    args:
        arrs:       (m, n) or (# of arrays, m, n)
        kernel:     (p, q) or (# of kernels, p, q)
        xp:         numpy or cupy
    returns:
        out:        (# of arrays, # of kernels, m - p + 1, n - q + 1)
    """
    arrs = arr
    kernels = kernel
    if arr.ndim == 2:
        arrs = arrs[np.newaxis]
    if kernel.ndim == 2:
        kernels = kernels[np.newaxis]
    n_arrs = arrs.shape[0]
    n_kernels = kernels.shape[0]
    n_rows = arrs.shape[1] - kernels.shape[1] + 1
    n_cols = arrs.shape[2] - kernels.shape[2] + 1
    out = xp.empty((n_arrs, n_kernels, n_rows, n_cols))
    for i, arr in enumerate(arrs):
        for j, kernel in enumerate(kernels):
            out[i, j] = convolve2d(arr, kernel, xp)
    return out


def dict_combinations(key_values: dict) -> list:
    """
    args:
        key_values:     {key0: [val0, val1, ...], ...}
    returns:
        [{key0: val0, ...}, {key0: val1, ...}, ...]
    """
    value_combinations = list(product(*key_values.values()))
    list_of_dicts = [
        dict(zip(key_values.keys(), values)) for values in value_combinations
    ]
    return list_of_dicts


def f_of_3_neighbors(u: np.ndarray, f: callable, xp: ModuleType) -> np.ndarray:
    """
    apply a function f (np.min or np.max) to each cell and it's 2 neighbors
    args:
        u:      (m,)
        f:      function of an array slice of u
        xp:     numpy or cupy
    returns:
        out:    (m - 2,)
    """
    list_of_3_neighbors = [
        u[..., 1:-1],
        u[..., :-2],
        u[..., 2:],
    ]
    return f(xp.asarray(list_of_3_neighbors), axis=0)


def f_of_5_neighbors(u: np.ndarray, f: callable, xp: ModuleType) -> np.ndarray:
    """
    apply a function f (np.min or np.max) to each cell and it's 4 neighbors
    args:
        u:      (m, n)
        f:      function of an array slice of u
        xp:     numpy or cupy
    returns:
        out:    (m - 2, n - 2)
    """
    list_of_5_neighbors = [
        u[..., 1:-1, 1:-1],
        u[..., :-2, 1:-1],
        u[..., 2:, 1:-1],
        u[..., 1:-1, :-2],
        u[..., 1:-1, 2:],
    ]
    return f(xp.asarray(list_of_5_neighbors), axis=0)


def f_of_9_neighbors(u: np.ndarray, f: callable, xp: ModuleType) -> np.ndarray:
    """
    apply a function f (np.min or np.max) to each cell and it's 8 neighbors
    args:
        u:      (m, n)
        f:      function of an array slice of u
        xp:     numpy or cupy
    returns:
        out:    (m - 2, n - 2)
    """
    list_of_9_neighbors = [
        u[..., 1:-1, 1:-1],
        u[..., :-2, 1:-1],
        u[..., 2:, 1:-1],
        u[..., 1:-1, :-2],
        u[..., 1:-1, 2:],
        u[..., 2:, 2:],
        u[..., 2:, :-2],
        u[..., :-2, 2:],
        u[..., :-2, :-2],
    ]
    return f(xp.asarray(list_of_9_neighbors), axis=0)


def np_floor(x: np.ndarray, floor: float) -> np.ndarray:
    """
    args:
        x:      any shape
        floor:  constant
    returns:
        x which doesn't subceed floor
    """
    return np.where(x < floor, floor, x)


def pad_uniform_extrap(x: np.ndarray, pad_width: int) -> np.ndarray:
    """
    args:
        x:          1D array of uniformly spaced values (m,)
        pad_width:  number of pad elements on either side
    returns:
        out:        1D array continuing the uniformly spaced array (m + 2 * pad_width,)
    """
    if pad_width == 0:
        return x
    h = np.mean(x[1:] - x[:-1])
    out = np.pad(x, pad_width)
    out[slice(None, pad_width)] = h * np.arange(-pad_width, 0) + x[0]
    out[slice(-pad_width, None)] = h * np.arange(1, pad_width + 1) + x[-1]
    return out


def RK_dt_adjust(h: float, spatial_order: int, temporal_order: int = 4) -> float:
    """
    args:
        h:                  finite volume discretization size
        spatial_order:      of accuracy
        temporal_order:     of accuracy
    returns:
        cfl_ajustment:      courant factor multiplier which makes nth order RK have
                            spatial_order order of accuracy
    """
    cfl_ajustment = h ** max((spatial_order - temporal_order) / temporal_order, 0)
    return cfl_ajustment


def quadrature_mesh(
    x: np.ndarray, y: np.ndarray, quadrature: np.ndarray, axis: int
) -> tuple:
    """
    args:
        x:              1d array (n,)
        y:              1d array (m,)
        quadrature:     1d array (k,), values in [-0.5, 0.5]
        axis:           0 or 1
    returns:
        xx:             xx spaced by quadratrue values along axis 0 if axis == 1
                        3d array (k, m, n)
        yy:             yy spaced by quadratrue values along axis 0 if axis == 0
                        3d array (k, m, n)
    """
    hx, hy = np.mean(x[1:] - x[:-1]), np.mean(y[1:] - y[:-1])
    xx, yy = np.meshgrid(x, y)
    na = np.newaxis
    xx = np.repeat(xx[na], len(quadrature), axis=0)
    yy = np.repeat(yy[na], len(quadrature), axis=0)
    if axis == 0:
        yy += quadrature[:, na, na] * hy
    elif axis == 1:
        xx += quadrature[:, na, na] * hx
    return xx, yy


def transpose_in_other_direction(x: np.ndarray) -> np.ndarray:
    """
    np.transpose but about the other axis
    """
    return np.fliplr(np.transpose(np.fliplr(x)))
