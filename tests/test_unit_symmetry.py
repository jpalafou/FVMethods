import pytest
import numpy as np
from finite_volume.utils import convolve_batch2d
from finite_volume.a_priori import mpp_limiter
from finite_volume.sed import compute_alpha_1d, compute_alpha_2d
from finite_volume.a_posteriori import (
    find_trouble,
    minmod,
    moncen,
    compute_MUSCL_interpolations_1d,
    compute_MUSCL_interpolations_2d,
    compute_PP2D_interpolations,
    broadcast_troubled_cells_to_faces_2d,
    broadcast_troubled_cells_to_faces_with_blending_2d,
)


def equivariant(
    f: callable,
    action: callable,
    fargs: dict,
    fargs_fixed: dict = None,
    out_argnums: list = None,
    invariant: bool = False,
    vnorm: str = "linf",
) -> float:
    """
    args:
        f               f(**fargs, **fargs_fixed)
        action          group action on f
        fargs           arguments of f which are subjected to the group action
        fargs_fixed     arguments of f which are not subjected to the group action
        out_argnums     int or list, returns f(**fargs, **fargs_fixed)[out_argnums0]...
        invariant       whether to check invariance
        vnorm           vector norm, 'l1', 'l2', 'linf'
    returns:
        error between
        f(action(**fargs), **fargs_fixed) and action(f(**fargs, **fargs_fixed))
        if invariant == False
        f(action(**fargs), **fargs_fixed) and f(**fargs, **fargs_fixed)
        if invariant == True
    """
    fargs_fixed = dict() if fargs_fixed is None else fargs_fixed

    if out_argnums is not None:

        def fout(**kwargs):
            argnums = (out_argnums,) if isinstance(out_argnums, int) else out_argnums
            out = f(**kwargs)
            for i in argnums:
                out = out[i]
            return out

    else:
        fout = f
    inner = fout(**{arg: action(value) for arg, value in fargs.items()}, **fargs_fixed)
    if not invariant:
        outer = action(fout(**fargs, **fargs_fixed))
    else:
        outer = fout(**fargs, **fargs_fixed)
    if isinstance(inner, np.ndarray):
        if vnorm == "l1":
            return np.mean(np.abs(inner - outer))
        elif vnorm == "l2":
            return np.sqrt(np.mean(np.square(inner - outer)))
        elif vnorm == "linf":
            return np.max(np.abs(inner - outer))
    # scalar
    return abs(inner - outer)


def linear_transformation(x, a, b):
    return a * x + b


def rotate(x, n: int):
    if isinstance(x, tuple):
        return tuple(rotate(i, n) for i in x)
    if x.ndim == 2:
        return np.rot90(x, n)
    elif x.ndim == 3:
        return np.rot90(x, n, axes=(1, 2))
    elif x.ndim == 4:
        return np.rot90(x, n, axes=(2, 3))


@pytest.mark.parametrize("n_test", range(10))
@pytest.mark.parametrize("arr_shape", [(20, 20), (14, 20, 20)])
@pytest.mark.parametrize("kernel_shape", [(10, 10), (2, 10, 10)])
@pytest.mark.parametrize("a", [-1, 1])
@pytest.mark.parametrize("b", [-2, -1, 0, 1, 2])
def test_convolve_batch2d_translation_equivariance(
    n_test, arr_shape, kernel_shape, a, b
):
    """
    convolve_batch2d(a * arr + b, kernel) == a * convolve_batch2d(arr, kernel) + b
    """
    arr = np.random.rand(*arr_shape)
    kernel = np.random.rand(*kernel_shape)
    # kernel must have unit weight
    if kernel.ndim == 2:
        kernel /= np.sum(kernel)
    elif kernel.ndim == 3:
        kernel /= np.sum(kernel, axis=(1, 2), keepdims=True)
    err = equivariant(
        convolve_batch2d,
        lambda x: linear_transformation(x, a, b),
        fargs=dict(arr=arr),
        fargs_fixed=dict(kernel=kernel),
        vnorm="l1",
    )
    assert err < 1e-14


@pytest.mark.parametrize("n_test", range(10))
@pytest.mark.parametrize("arr_shape", [(20, 20), (14, 20, 20)])
@pytest.mark.parametrize("kernel_shape", [(1, 10), (10, 10), (2, 10, 10)])
@pytest.mark.parametrize("n_rotations", [0, 1, 2, 3])
def test_convolve_batch2d_rotation_equivariance(
    n_test, arr_shape, kernel_shape, n_rotations
):
    """
    convolve_batch2d(rotate(arr), rotate(kernel))
    == rotate(convolve_batch2d(arr, kernel))
    """
    arr = np.random.rand(*arr_shape)
    kernel = np.random.rand(*kernel_shape)

    # kernel must have unit weight
    if kernel.ndim == 2:
        kernel /= np.sum(kernel)
    elif kernel.ndim == 3:
        kernel /= np.sum(kernel, axis=(1, 2), keepdims=True)
    err = equivariant(
        convolve_batch2d,
        lambda x: rotate(x, n=n_rotations),
        fargs=dict(arr=arr, kernel=kernel),
        vnorm="l1",
    )
    assert err < 1e-14


@pytest.mark.parametrize("n_test", range(200))
@pytest.mark.parametrize("a", [-1, 1])
@pytest.mark.parametrize("b", [-2, -1, 0, 1, 2])
def test_compute_alpha_1d_translation_invariance(n_test, a, b):
    """
    compute_alpha_1d(a * u + b) == compute_alpha_1d(u)
    """
    u = np.random.rand(64)
    err = equivariant(
        compute_alpha_1d,
        lambda x: linear_transformation(x, a, b),
        fargs=dict(u=u),
        invariant=True,
        vnorm="l1",
    )
    assert err < 1e-14


@pytest.mark.parametrize("n_test", range(10))
@pytest.mark.parametrize("a", [-1, 1])
@pytest.mark.parametrize("b", [-2, -1, 0, 1, 2])
def test_compute_alpha_2d_translation_invariance(n_test, a, b):
    """
    compute_alpha_2d(a * u + b) == compute_alpha_2d(u)
    """
    u = np.random.rand(64, 64)
    err = equivariant(
        compute_alpha_2d,
        lambda x: linear_transformation(x, a, b),
        fargs=dict(u=u),
        invariant=True,
        vnorm="l1",
    )
    assert err < 1e-14


@pytest.mark.parametrize("n_test", range(10))
@pytest.mark.parametrize("n_rotations", [0, 1, 2, 3])
def test_compute_alpha_2d_rotation_equivariance(n_test, n_rotations):
    """
    compute_alpha_2d(rotate(u)) == rotate(compute_alpha_2d(u))
    """
    u = np.random.rand(64, 64)
    err = equivariant(
        compute_alpha_2d,
        lambda x: rotate(x, n=n_rotations),
        fargs=dict(u=u),
        vnorm="l1",
    )
    assert err < 1e-14


@pytest.mark.parametrize("n_test", range(10))
@pytest.mark.parametrize(
    "shapes",
    [
        ((66,), (1, 64)),
        ((66,), (10, 64)),
        ((66, 130), (1, 1, 64, 128)),
        ((66, 130), (10, 10, 64, 128)),
    ],
)
@pytest.mark.parametrize("a", [-1, 1])
@pytest.mark.parametrize("b", [-2, -1, 0, 1, 2])
def test_mpp_limiter_translation_invariance(n_test, shapes, a, b):
    """
    mpp_limiter(a * u + b, a * points + b) == mpp_limiter(u, points)
    """
    u = np.random.rand(*shapes[0])
    points = np.random.rand(*shapes[1])
    err = equivariant(
        mpp_limiter,
        lambda x: linear_transformation(x, a, b),
        out_argnums=0,
        fargs=dict(u=u, points=points),
        invariant=True,
        vnorm="l1",
    )
    assert err < 1e-14


@pytest.mark.parametrize("n_test", range(10))
@pytest.mark.parametrize(
    "shapes",
    [
        ((66, 130), (1, 1, 64, 128)),
        ((66, 130), (10, 10, 64, 128)),
    ],
)
@pytest.mark.parametrize("n_rotations", [0, 1, 2, 3])
def test_mpp_limiter_rotation_equivariance(n_test, shapes, n_rotations):
    """
    mpp_limiter(rotate(u), rotate(points)) == rotate(mpp_limiter(u, points))
    """
    u = np.random.rand(*shapes[0])
    points = np.random.rand(*shapes[1])
    err = equivariant(
        mpp_limiter,
        lambda x: rotate(x, n=n_rotations),
        out_argnums=0,
        fargs=dict(u=u, points=points),
        vnorm="l1",
    )
    assert err < 1e-14


@pytest.mark.parametrize("n_test", range(10))
@pytest.mark.parametrize(
    "shapes",
    [
        ((34,), (38,)),
        ((34, 66), (38, 70)),
    ],
)
@pytest.mark.parametrize("NAD", [0.0, 1e-10, 1e-3])
@pytest.mark.parametrize("PAD", [np.array((-np.inf, np.inf)), np.array((0, 1))])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("a", [-1, 1])
@pytest.mark.parametrize("b", [-2, -1, 0, 1, 2])
def test_find_trouble_translation_invariance(n_test, shapes, NAD, PAD, SED, a, b):
    """
    find_trouble(a * u + b, a * u_candidate + b) == find_trouble(u, u_candidate)
    """
    u = np.random.rand(*shapes[0])
    u_candidate = np.random.rand(*shapes[1])
    err = equivariant(
        f=find_trouble,
        action=lambda x: linear_transformation(x, a, b),
        fargs=dict(u=u, u_candidate=u_candidate, PAD=PAD),
        fargs_fixed=dict(NAD=NAD, SED=SED),
        out_argnums=0,
        invariant=True,
        vnorm="linf",
    )
    assert err == 0


@pytest.mark.parametrize("n_test", range(10))
@pytest.mark.parametrize("NAD", [0.0, 1e-10, 1e-3])
@pytest.mark.parametrize("PAD", [np.array((-np.inf, np.inf)), np.array((0, 1))])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("n_rotations", [0, 1, 2, 3])
def test_find_trouble_rotation_equivariance(n_test, NAD, PAD, SED, n_rotations):
    """
    find_trouble(rotate(u), rotate(u_candidate)) == rotate(find_trouble(u, u_candidate))
    """
    u = np.random.rand(34, 66)
    u_candidate = np.random.rand(38, 70)
    err = equivariant(
        f=find_trouble,
        action=lambda x: rotate(x, n=n_rotations),
        fargs=dict(u=u, u_candidate=u_candidate),
        fargs_fixed=dict(NAD=NAD, SED=SED, PAD=PAD),
        out_argnums=0,
        vnorm="linf",
    )
    assert err == 0


@pytest.mark.parametrize("n_test", range(10))
@pytest.mark.parametrize("slope_limiter", [minmod, moncen])
@pytest.mark.parametrize("a", [1, -1])
@pytest.mark.parametrize("b", [-2, -1, 0, 1, 2])
def test_minmod_moncen_translation_invariance(n_test, slope_limiter, a, b):
    """
    slopelim(a * du_left + b, a * du_right + b) == a * slopelim(du_left, du_right) + b
    """
    u = np.random.rand(32, 66)

    def limiter(u):
        du_left = u[:, 1:-1] - u[:, :-2]
        du_right = u[:, 2:] - u[:, 1:-1]
        return slope_limiter(du_left, du_right)

    if a == 1:
        err = equivariant(
            f=limiter,
            action=lambda x: linear_transformation(x, a, b),
            fargs=dict(u=u),
            invariant=True,
            vnorm="l1",
        )
        assert err < 1e-14
    elif a == -1 and b == 0:
        err = equivariant(
            f=limiter,
            action=lambda x: linear_transformation(x, a, b),
            fargs=dict(u=u),
            vnorm="l1",
        )
        assert err < 1e-14
    # null case


@pytest.mark.parametrize("n_test", range(10))
@pytest.mark.parametrize("slope_limiter", [minmod, moncen])
@pytest.mark.parametrize("fallback_to_1st_order", [False, True])
@pytest.mark.parametrize("PAD", [np.array((-np.inf, np.inf)), np.array((0, 1))])
@pytest.mark.parametrize("hancock", [False, True])
@pytest.mark.parametrize("a", [1])
@pytest.mark.parametrize("b", [-2, -1, 0, 1, 2])
def test_compute_MUSCL_interpolations_1d_translation_equivariance(
    n_test, slope_limiter, fallback_to_1st_order, PAD, hancock, a, b
):
    """
    compute_MUSCL_interpolations_1d(a * u + b)
    == a * compute_MUSCL_interpolations_1d(u) + b
    """
    u = np.random.rand(34)
    v = np.random.rand(32) - 0.5
    err = equivariant(
        f=compute_MUSCL_interpolations_1d,
        action=lambda x: linear_transformation(x, a, b),
        out_argnums=0,
        fargs=dict(u=u, PAD=PAD),
        fargs_fixed=dict(
            slope_limiter=slope_limiter,
            fallback_to_1st_order=fallback_to_1st_order,
            hancock=hancock,
            dt=0.01,
            h=0.02,
            v_cell_centers=v,
        ),
        vnorm="l1",
    )
    assert err < 1e-14


@pytest.mark.parametrize("n_test", range(10))
@pytest.mark.parametrize("slope_limiter", [minmod, moncen])
@pytest.mark.parametrize("fallback_to_1st_order", [False, True])
@pytest.mark.parametrize("PAD", [np.array((-np.inf, np.inf)), np.array((0, 1))])
@pytest.mark.parametrize("hancock", [False, True])
@pytest.mark.parametrize("a", [1])
@pytest.mark.parametrize("b", [-2, -1, 0, 1, 2])
def test_compute_MUSCL_interpolations_2d_translation_equivariance(
    n_test, slope_limiter, fallback_to_1st_order, PAD, hancock, a, b
):
    """
    compute_MUSCL_interpolations_2d(a * u + b)
    == a * compute_MUSCL_interpolations_2d(u) + b
    """
    u = np.random.rand(34, 66)
    vx = np.random.rand(32, 64) - 0.5
    vy = np.random.rand(32, 64) - 0.5
    err = equivariant(
        f=compute_MUSCL_interpolations_2d,
        action=lambda x: linear_transformation(x, a, b),
        out_argnums=(0, 0),
        fargs=dict(u=u, PAD=PAD),
        fargs_fixed=dict(
            slope_limiter=slope_limiter,
            fallback_to_1st_order=fallback_to_1st_order,
            hancock=hancock,
            dt=0.01,
            h=(0.02, 0.02),
            v_cell_centers=(vx, vy),
        ),
        vnorm="l1",
    )
    assert err < 1e-14


@pytest.mark.parametrize("n_test", range(10))
@pytest.mark.parametrize("slope_limiter", [minmod, moncen])
@pytest.mark.parametrize("fallback_to_1st_order", [False, True])
@pytest.mark.parametrize("PAD", [np.array((-np.inf, np.inf)), np.array((0, 1))])
@pytest.mark.parametrize("hancock", [False, True])
@pytest.mark.parametrize("n_rotations", [0, 1, 2, 3])
def test_compute_MUSCL_interpolations_2d_rotation_equivariance(
    n_test, slope_limiter, fallback_to_1st_order, PAD, hancock, n_rotations
):
    """
    compute_MUSCL_interpolations_2d(rotate(u))
    == rotate(compute_MUSCL_interpolations_2d(u))
    """
    u = np.random.rand(34, 66)
    vx = np.random.rand(32, 64) - 0.5
    vy = np.random.rand(32, 64) - 0.5
    config = dict(
        slope_limiter=slope_limiter,
        fallback_to_1st_order=fallback_to_1st_order,
        hancock=hancock,
        dt=0.01,
        h=(0.02, 0.02),
        PAD=PAD,
    )
    idxs = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}  # west south east north
    vs = {
        0: (vx, vy),
        1: (rotate(vy, 1), -rotate(vx, 1)),
        2: (-rotate(vx, 2), -rotate(vy, 2)),
        3: (-rotate(vy, 3), rotate(vx, 3)),
    }
    inner = compute_MUSCL_interpolations_2d(
        rotate(u, n_rotations), v_cell_centers=vs[n_rotations], **config
    )[0][0]
    outer = rotate(
        compute_MUSCL_interpolations_2d(u, v_cell_centers=(vx, vy), **config),
        n_rotations,
    )[idxs[n_rotations][0]][idxs[n_rotations][1]]
    err = np.mean(np.abs(inner - outer))
    assert err < 1e-14


@pytest.mark.parametrize("n_test", range(10))
@pytest.mark.parametrize("hancock", [False, True])
@pytest.mark.parametrize("a", [1])
@pytest.mark.parametrize("b", [-2, -1, 0, 1, 2])
def test_compute_PP2D_interpolations_translation_equivariance(n_test, hancock, a, b):
    """
    compute_MUSCL_interpolations_2d(a * u + b)
    == a * compute_MUSCL_interpolations_2d(u) + b
    """
    u = np.random.rand(34, 66)
    vx = np.random.rand(32, 64) - 0.5
    vy = np.random.rand(32, 64) - 0.5
    err = equivariant(
        f=compute_PP2D_interpolations,
        action=lambda x: linear_transformation(x, a, b),
        out_argnums=(0, 0),
        fargs=dict(u=u),
        fargs_fixed=dict(
            hancock=hancock,
            dt=0.01,
            h=(0.02, 0.02),
            v_cell_centers=(vx, vy),
        ),
        vnorm="l1",
    )
    assert err < 1e-14


@pytest.mark.parametrize("n_test", range(10))
@pytest.mark.parametrize("hancock", [False, True])
@pytest.mark.parametrize("n_rotations", [0, 1, 2, 3])
def test_compute_PP2D_interpolations_rotation_equivariance(
    n_test, hancock, n_rotations
):
    """
    compute_MUSCL_interpolations_2d(rotate(u))
    == rotate(compute_MUSCL_interpolations_2d(u))
    """
    u = np.random.rand(34, 66)
    vx = np.random.rand(32, 64) - 0.5
    vy = np.random.rand(32, 64) - 0.5
    config = dict(
        hancock=hancock,
        dt=0.01,
        h=(0.02, 0.02),
    )
    idxs = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}  # west south east north
    vs = {
        0: (vx, vy),
        1: (rotate(vy, 1), -rotate(vx, 1)),
        2: (-rotate(vx, 2), -rotate(vy, 2)),
        3: (-rotate(vy, 3), rotate(vx, 3)),
    }
    inner = compute_PP2D_interpolations(
        rotate(u, n_rotations), v_cell_centers=vs[n_rotations], **config
    )[0][0]
    outer = rotate(
        compute_PP2D_interpolations(u, v_cell_centers=(vx, vy), **config),
        n_rotations,
    )[idxs[n_rotations][0]][idxs[n_rotations][1]]
    err = np.mean(np.abs(inner - outer))
    assert err < 1e-14


@pytest.mark.parametrize("n_test", range(20))
@pytest.mark.parametrize("n_rotations", [0, 1, 2, 3])
def test_broadcast_troubled_cells_to_faces_2d_rotation_equivariance(
    n_test, n_rotations
):
    """
    broadcast_troubled_cells_to_faces_2d(rotation(u))
    == rotation(broadcast_troubled_cells_to_faces_2d(u))
    """
    trouble = np.random.randint(0, 2, (100, 100))
    inner = broadcast_troubled_cells_to_faces_2d(rotate(trouble, n_rotations))[0]
    outer = rotate(broadcast_troubled_cells_to_faces_2d(trouble), n_rotations)[
        n_rotations % 2 != 0
    ]
    assert np.max(np.abs(inner - outer)) == 0


@pytest.mark.parametrize("n_test", range(20))
@pytest.mark.parametrize("n_rotations", [0, 1, 2, 3])
def test_broadcast_troubled_cells_to_faces_with_blending_2d_rotation_equivariance(
    n_test, n_rotations
):
    """
    broadcast_troubled_cells_to_faces_with_blending(rotation(u))
    == broadcast_troubled_cells_to_faces_with_blending(rotation(u))
    """
    trouble = np.random.randint(0, 2, (100, 100))
    inner = broadcast_troubled_cells_to_faces_with_blending_2d(
        rotate(trouble, n_rotations)
    )[0]
    outer = rotate(
        broadcast_troubled_cells_to_faces_with_blending_2d(trouble), n_rotations
    )[n_rotations % 2 != 0]
    assert np.max(np.abs(inner - outer)) == 0
