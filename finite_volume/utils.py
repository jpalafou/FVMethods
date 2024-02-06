from itertools import product
from numba import njit, prange
import numpy as np


def rk4_dt_adjust(n: int, spatial_order: int) -> float:
    """
    args:
        n:              number of cells
        spatial_order:  of accuracy
    returns:
        courant factor which makes rk4 have the same order of accuracy as spatial_order
    """
    return (1 / n) ** max((spatial_order - 4) / 4, 0)


def stack(u: np.ndarray, stacks: int, axis: int = 0):
    """
    args:
        u       array of arbitrary shape
        stacks  number of stacks to form
        axis
    returns:
        array with a new first axis of length (stacks)
        and (stacks - 1) reduced length along (axis + 1)
    """
    shape = list(u.shape)
    shape[axis] -= stacks - 1
    out = np.concatenate(
        [
            np.expand_dims(u.take(range(i, i + shape[axis]), axis=axis), axis=0)
            for i in range(stacks)
        ],
        axis=0,
    )
    return out


def apply_stencil(u: np.ndarray, stencil: np.ndarray, axis: int = 0):
    """
    args:
        u       array of arbitrary shape
        stencil 1d np array
        axis
    returns:
        stensil weighted sum of neighbors
        with (stensil.size - 1) reduced length along (axis)
    """
    new_stencil_shape = np.ones(u.ndim, dtype=int)
    new_stencil_shape[axis] = -1
    reshaped_stencil = stencil.reshape(new_stencil_shape)
    return np.sum(u * reshaped_stencil, axis=axis) / np.sum(stencil)


def chop(u, chop_size, axis):
    """
    symmetric chop of array edges about axis
    args:
        u           np array of arbitrary dimension
        chop_size   int
        axis        int or list of axes
    returns:
        u           symmetrically cut at the ends along axis by cut_length
    """
    index = np.array([slice(None)] * u.ndim)
    index[axis] = slice(chop_size, -chop_size or None)
    return u[tuple(index)]


def chopchop(u, chop_size, axis):
    """
    asymmetric chop of array edges about axis
    args:
        u           np array of arbitrary dimension
        chop_size   tuple
        axis        int or list of axes
    returns:
        u           symmetrically cut at the ends along axis by cut_length
    """
    index = np.array([slice(None)] * u.ndim)
    index[axis] = slice(chop_size[0], -chop_size[1] or None)
    return u[tuple(index)]


def f_of_3_neighbors(u: np.array, f):
    """
    apply a function f (np.minimum or np.maximum) to each cell and it's 2 neighbors
    args:
        u   (m,)
    returns:
        out (m - 2,)
    """
    list_of_3_neighbors = [
        u[..., 1:-1],
        u[..., :-2],
        u[..., 2:],
    ]
    return f.reduce(list_of_3_neighbors)


def f_of_4_neighbors(u: np.array, f):
    """
    apply a function f (np.minimum or np.maximum) to each cell and it's 8 neighbors
    args:
        u   (m, n)
    returns:
        out (m - 2, n - 2)
    """
    list_of_9_neighbors = [
        u[..., 1:-1, 1:-1],
        u[..., :-2, 1:-1],
        u[..., 2:, 1:-1],
        u[..., 1:-1, :-2],
        u[..., 1:-1, 2:],
    ]
    return f.reduce(list_of_9_neighbors)


def f_of_9_neighbors(u: np.array, f):
    """
    apply a function f (np.minimum or np.maximum) to each cell and it's 8 neighbors
    args:
        u   (m, n)
    returns:
        out (m - 2, n - 2)
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
    return f.reduce(list_of_9_neighbors)


def dict_combinations(key_values: dict) -> list:
    """
    args:
        key_values  {key0: [val0, val1, ...], ...}
    returns:
        [{key0: val0, ...}, {key0: val1, ...}, ...]
    """
    value_combinations = list(product(*key_values.values()))
    list_of_dicts = [
        dict(zip(key_values.keys(), values)) for values in value_combinations
    ]
    return list_of_dicts


@njit(parallel=True)
def batch_convolve2d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    args:
        arrs    (m, n) or (# of arrays, m, n)
        kernel  (p, q) or (# of kernels, p, q)
    returns:
        out     (# of arrays, # of kernels, m - p + 1, n - q + 1)
    """
    # add extra first axis where necessary
    arrs = arr[np.newaxis, ...] if arr.ndim == 2 else arr
    kernels = kernel[np.newaxis, ...] if kernel.ndim == 2 else kernel

    # get array shapes
    n_arrays, arr_rows, arr_cols = arrs.shape
    n_kernels, kern_rows, kern_cols = kernels.shape

    # intialize empty array
    out = np.empty(
        (n_arrays, n_kernels, arr_rows - kern_rows + 1, arr_cols - kern_cols + 1),
        dtype=np.float64,
    )

    # perform n_arrays * n_kernels convolutions
    for i in prange(n_arrays):
        for j in prange(n_kernels):
            out[i, j, ...] = convolve2d(arrs[i, ...], kernels[j, ...])

    return out


@njit
def convolve2d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    args:
        arr     2D array
        kernel  2D array
    returns:
        out     arr convolved with kernel, excluding boundaries
    """
    # get array shapes
    arr_rows, arr_cols = arr.shape
    kern_rows, kern_cols = kernel.shape

    # initialize empty array
    out = np.empty(
        (arr_rows - kern_rows + 1, arr_cols - kern_cols + 1), dtype=np.float64
    )
    out_rows, out_cols = out.shape

    # perform convolution
    for i in range(out_rows):
        for j in range(out_cols):
            value = 0.0
            for p in range(kern_rows):
                for q in range(kern_cols):
                    value += arr[i + p, j + q] * kernel[p, q]
            out[i, j] = value

    return out


def np_floor(x: np.ndarray, floor: float) -> np.ndarray:
    """
    args:
        x       any shape
        floor   constant
    returns:
        x which doesn't subceed floor
    """
    return np.where(x < floor, floor, x)
