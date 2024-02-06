import pytest
import numpy as np
from finite_volume.utils import batch_convolve2d
from finite_volume.a_priori import mpp_limiter


def equivariant(
    f: callable,
    action: callable,
    fargs: dict,
    fargs_fixed: dict = None,
    atol: float = 0.0,
    rtol: float = 0.0,
):
    """
    args:
        f               f(**fargs, **fargs_fixed)
        action          group action on f
        fargs           arguments of f
        fargs_fixed     arguments of f
        atol            absolute tolerance
        rtol            relative tolerance
    returns:
        f(action(**fargs), **fargs_fixed) == action(f(**fargs, **fargs_fixed))
    """
    inner = f(**{arg: action(value) for arg, value in fargs.items()}, **fargs_fixed)
    outer = action(f(**fargs, **fargs_fixed))
    if isinstance(inner, np.ndarray):
        # numpy
        return np.all(np.isclose(inner, outer, atol=atol, rtol=rtol))
    # scalar
    return abs(inner - outer) <= atol + rtol * abs(outer)


def linear_transformation(x, a, b):
    return a * x + b


@pytest.mark.parametrize("n_test", range(5))
@pytest.mark.parametrize("arr_shape", [(100, 100), (14, 100, 100)])
@pytest.mark.parametrize("kernel_shape", [(10, 10), (20, 10, 10)])
@pytest.mark.parametrize("a", [-1, 1])
@pytest.mark.parametrize("b", [-2, -1, 0, 1, 2])
def test_batch_convolve2d_equivariance(n_test, arr_shape, kernel_shape, a, b):
    """
    batch_convolve2d(a * arr + b, kernel) == a * batch_convolve2d(arr, kernel) + b
    """
    arr = np.random.rand(*arr_shape)
    print(arr.shape)
    kernel = np.random.rand(*kernel_shape)
    # kernel must have unit weight
    if kernel.ndim == 2:
        kernel /= np.sum(kernel)
    elif kernel.ndim == 3:
        kernel /= np.sum(kernel, axis=(1, 2), keepdims=True)
    assert equivariant(
        batch_convolve2d,
        lambda x: linear_transformation(x, a, b),
        fargs=dict(arr=arr),
        fargs_fixed=dict(kernel=kernel),
        atol=1e-14,
    )
