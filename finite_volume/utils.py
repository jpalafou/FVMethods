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
