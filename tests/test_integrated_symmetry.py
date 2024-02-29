import pytest
import numpy as np
import os
from finite_volume.advection import AdvectionSolver
from finite_volume.initial_conditions import generate_ic
from finite_volume.utils import dict_combinations, transpose_in_other_direction

test_directory = "data/test_solutions/"


@pytest.fixture(scope="module", autouse=True)
def cleanup(request):
    """
    Remove solutions saved by tests
    """

    def remove_test_dir():
        for f in os.listdir(test_directory):
            os.remove(os.path.join(test_directory, f))
        os.rmdir(test_directory)

    request.addfinalizer(remove_test_dir)


def l1(x: np.ndarray) -> float:
    return np.mean(np.abs(x))


def l2(x: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(x)))


def linf(x: np.ndarray) -> float:
    return np.max(np.abs(x))


"""
1D
"""

limiter_configs_1d = [
    dict(),
    *dict_combinations(
        dict(
            apriori_limiting=[True],
            mpp_lite=[False, True],
        )
    ),
    *dict_combinations(
        dict(
            aposteriori_limiting=[True],
            fallback_limiter=["minmod", "moncen"],
            convex=[False, True],
            hancock=[True],
            fallback_to_first_order=[False, True],
        )
    ),
]


@pytest.mark.parametrize("p", range(8))
@pytest.mark.parametrize("a", [1, -1])
@pytest.mark.parametrize("k", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("ic_type__PAD", [("sinus", (-1, 1)), ("square", (0, 1))])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("limiter_config", limiter_configs_1d)
def test_translation_equivariance_1d(p, a, k, ic_type__PAD, SED, limiter_config):
    """
    advection_solution(a * u(x) + k) = a * advection_solution(u(x)) + k
    """
    ic_type, PAD = ic_type__PAD

    def linear_transformation(x):
        return a * x + k

    def u0(x):
        return generate_ic(type=ic_type, x=x, y=None)

    def u0_shifted(x):
        return linear_transformation(u0(x))

    u0_shifted.__name__ = u0_shifted.__name__ + f"_{a}_{k}"

    shared_config = dict(
        **limiter_config,
        save_directory=test_directory,
        v=1,
        n=64,
        order=p + 1,
        courant=0.8,
        snapshot_dt=1,
        SED=SED,
    )

    # baseline
    solver = AdvectionSolver(
        **shared_config,
        u0=u0,
        PAD=PAD,
    )
    solver.rkorder()

    # shifted initial condition
    print(sorted((linear_transformation(PAD[0]), linear_transformation(PAD[1]))))
    translated_solver = AdvectionSolver(
        **shared_config,
        u0=u0_shifted,
        PAD=sorted((linear_transformation(PAD[0]), linear_transformation(PAD[1]))),
    )
    translated_solver.rkorder()

    # check equivariance
    diffs = (
        linear_transformation(solver.u_snapshots[-1][1])
        - translated_solver.u_snapshots[-1][1]
    )
    print(f"{l1(diffs)=}")
    print(f"{l2(diffs)=}")
    print(f"{linf(diffs)=}")
    err = l2(diffs)
    assert err < 1e-14


@pytest.mark.parametrize("p", range(8))
@pytest.mark.parametrize("ic_type__PAD", [("sinus", (-1, 1)), ("square", (0, 1))])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("limiter_config", limiter_configs_1d)
def test_velocity_equivariance_1d(p, ic_type__PAD, SED, limiter_config):
    """
    advection_solution(u(x), -v_x) = advection_solution(u(-x), v_x)
    """
    ic_type, PAD = ic_type__PAD

    def u0(x):
        return generate_ic(type=ic_type, x=x, y=None)

    def u0_horizontally_reflected(x):
        return np.flip(u0(x))

    shared_config = dict(
        **limiter_config,
        save_directory=test_directory,
        PAD=PAD,
        n=64,
        order=p + 1,
        courant=0.8,
        snapshot_dt=1,
        SED=SED,
    )

    # baseline
    solver = AdvectionSolver(
        **shared_config,
        u0=u0,
        v=1,
    )
    solver.rkorder()

    # reflected
    solver_negative_velocity = AdvectionSolver(
        **shared_config,
        u0=u0_horizontally_reflected,
        v=-1,
    )
    solver_negative_velocity.rkorder()

    # check equivariance
    diffs = solver.u_snapshots[-1][1] - np.flip(
        solver_negative_velocity.u_snapshots[-1][1]
    )

    print(f"{l1(diffs)=}")
    print(f"{l2(diffs)=}")
    print(f"{linf(diffs)=}")
    err = l2(diffs)
    assert err < 1e-14


"""
2D
"""

limiter_configs_2d = [
    dict(),
    *dict_combinations(
        dict(
            apriori_limiting=[True],
            mpp_lite=[False, True],
        )
    ),
    *dict_combinations(
        dict(
            aposteriori_limiting=[True],
            fallback_limiter=["minmod", "moncen"],
            convex=[False, True],
            hancock=[True],
            fallback_to_first_order=[False, True],
        )
    ),
    *dict_combinations(
        dict(
            aposteriori_limiting=[True],
            fallback_limiter=["PP2D"],
            convex=[False, True],
            hancock=[True],
            fallback_to_first_order=[False],
        )
    ),
]


@pytest.mark.parametrize("p", range(8))
@pytest.mark.parametrize("a", [1, -1])
@pytest.mark.parametrize("k", [-2, 0, 2])
@pytest.mark.parametrize("ic_type__PAD", [("sinus", (-1, 1)), ("square", (0, 1))])
@pytest.mark.parametrize("quadrature", ["gauss-legendre", "transverse"])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("limiter_config", limiter_configs_2d[:1])
def test_translation_equivariance_2d(
    p, a, k, ic_type__PAD, quadrature, SED, limiter_config
):
    """
    advection_solution(a * u(x, y) + k) = a * advection_solution(u(x, y)) + k
    """
    ic_type, PAD = ic_type__PAD

    def linear_transformation(x):
        return a * x + k

    def u0(x, y):
        return generate_ic(type=ic_type, x=x, y=y)

    def u0_shifted(x, y):
        return linear_transformation(u0(x, y))

    u0_shifted.__name__ += f"_{a}_{k}"

    shared_config = dict(
        **limiter_config,
        save_directory=test_directory,
        v=(2, 1),
        n=(64,),
        order=p + 1,
        courant=0.8,
        snapshot_dt=1,
        flux_strategy=quadrature,
        SED=SED,
    )

    # baseline
    solver = AdvectionSolver(
        **shared_config,
        u0=u0,
        PAD=PAD,
    )
    solver.rkorder()

    # shifted initial condition
    translated_solver = AdvectionSolver(
        **shared_config,
        u0=u0_shifted,
        PAD=sorted((linear_transformation(PAD[0]), linear_transformation(PAD[1]))),
    )
    translated_solver.rkorder()

    # check equivariance
    diffs = (
        linear_transformation(solver.u_snapshots[-1][1])
        - translated_solver.u_snapshots[-1][1]
    )
    print(f"{l1(diffs)=}")
    print(f"{l2(diffs)=}")
    print(f"{linf(diffs)=}")
    err = l2(diffs)
    assert err < 1e-14


@pytest.mark.parametrize("p", range(8))
@pytest.mark.parametrize("ic_type__PAD", [("sinus", (-1, 1)), ("square", (0, 1))])
@pytest.mark.parametrize(
    "transformation", [np.fliplr, np.flipud, "rotate 1", "rotate 2", "rotate 3"]
)
@pytest.mark.parametrize("quadrature", ["gauss-legendre", "transverse"])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("limiter_config", limiter_configs_2d[:1])
def test_velocity_equivariance_2d(
    p, ic_type__PAD, transformation, quadrature, SED, limiter_config
):
    """
    for f in
        reflection about x=0
        reflection about y=0
        rotation by 90 degrees
        rotation by 180 degrees
        rotation by 270 degrees
    advection_solution(f(u(x, y)), f(v)) = f(advection_solution(u(x, y), f(v)))
    """
    ic_type, PAD = ic_type__PAD

    def u0(x, y):
        # offset to avoid inherent rotational symmetry
        return generate_ic(type=ic_type, x=x - 0.2, y=y - 0.1)

    if transformation == np.fliplr:
        f = transformation
        v_outer = (1, 0)
        v_inner = (-1, 0)
    elif transformation == np.flipud:
        f = transformation
        v_outer = (0, 1)
        v_inner = (0, -1)
    elif isinstance(transformation, str) and transformation[:7] == "rotate ":
        i = int(transformation.split(" ")[1])

        def f(x):
            return np.flipud(np.rot90(np.flipud(x), i))

        f.__name__ = f"rotation_{i}"
        if i == 1:
            v_outer = (1, 0)
            v_inner = (0, 1)
        elif i == 2:
            v_outer = (1, 0)
            v_inner = (-1, 0)
        elif i == 3:
            v_outer = (1, 0)
            v_inner = (0, -1)

    def u0_inner(x, y):
        return f(u0(x, y))

    u0_inner.__name__ += "_" + f.__name__

    shared_config = dict(
        **limiter_config,
        save_directory=test_directory,
        PAD=PAD,
        NAD=1e-5,
        n=(64,),
        order=p + 1,
        flux_strategy=quadrature,
        courant=0.8,
        snapshot_dt=1,
        SED=SED,
    )

    # baseline
    solver_outer = AdvectionSolver(
        **shared_config,
        u0=u0,
        v=v_outer,
    )
    solver_outer.rkorder()

    # reflected
    solver_inner = AdvectionSolver(
        **shared_config,
        u0=u0_inner,
        v=v_inner,
    )
    solver_inner.rkorder()

    # check equivariance
    outer = f(solver_outer.u_snapshots[-1][1])
    inner = solver_inner.u_snapshots[-1][1]
    diffs = inner - outer
    print(f"{l1(diffs)=}")
    print(f"{l2(diffs)=}")
    print(f"{linf(diffs)=}")
    err = l2(diffs)
    assert err < 1e-14


@pytest.mark.parametrize("p", range(8))
@pytest.mark.parametrize("ic_type__PAD", [("square", (0, 1))])
@pytest.mark.parametrize(
    "v",
    [
        (np.sqrt(2), 0.0),
        (1, 1),
        (0, np.sqrt(2)),
        (-1, 1),
        (-np.sqrt(2), 0),
        (-1, -1),
        (0, -np.sqrt(2)),
        (1, -1),
    ],
)
@pytest.mark.parametrize("quadrature", ["gauss-legendre", "transverse"])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("limiter_config", limiter_configs_2d[:1])
def test_reflection_equivariance_2d(
    p, ic_type__PAD, v, quadrature, SED, limiter_config
):
    """
    test symmetry of solution about
        y=0
        y=x
        x=0
        y=-x
    """
    ic_type, PAD = ic_type__PAD

    def u0(x, y):
        # offset to avoid inherent rotational symmetry
        return generate_ic(type=ic_type, x=x, y=y)

    if v in {(np.sqrt(2), 0.0), (-np.sqrt(2), 0)}:
        f = np.flipud
    elif v in {(1, 1), (-1, -1)}:

        def f(x):
            return np.flipud(transpose_in_other_direction(np.flipud(x)))

    elif v in {(0, np.sqrt(2)), (0, -np.sqrt(2))}:
        f = np.fliplr
    elif v in {(-1, 1), (1, -1)}:

        def f(x):
            return np.flipud(np.transpose(np.flipud(x)))

    shared_config = dict(
        **limiter_config,
        save_directory=test_directory,
        PAD=PAD,
        NAD=1e-5,
        n=(64,),
        order=p + 1,
        flux_strategy=quadrature,
        courant=0.8,
        snapshot_dt=1,
        SED=SED,
    )

    solver_outer = AdvectionSolver(
        **shared_config,
        u0=u0,
        v=v,
    )
    solver_outer.rkorder()

    # check reflection equivariance
    solution = solver_outer.u_snapshots[-1][1]
    reflected_solution = f(solution)
    diffs = solution - reflected_solution
    print(f"{l1(diffs)=}")
    print(f"{l2(diffs)=}")
    print(f"{linf(diffs)=}")
    err = l2(diffs)
    assert err < 1e-14


@pytest.mark.parametrize("p", range(8))
@pytest.mark.parametrize("quadrature", ["gauss-legendre", "transverse"])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("limiter_config", limiter_configs_2d[:1])
def test_disk_slotted_disk_velocity_equivariance(p, quadrature, SED, limiter_config):
    """
    test equivariance of slotted disk rotated counterclockwise and clockwise
    """

    def v_ccw(x, y):
        """
        counterclockwise vortex about (0, 0)
        """
        return -y, x

    def v_cw(x, y):
        """
        clockwise vortex about (0, 0)
        """
        return y, -x

    shared_config = dict(
        **limiter_config,
        save_directory=test_directory,
        PAD=(0, 1),
        NAD=1e-5,
        n=(128,),
        u0="disk",
        x=(-1, 1),
        order=p + 1,
        flux_strategy=quadrature,
        courant=0.8,
        snapshot_dt=2 * np.pi,
        SED=SED,
    )

    solver_cw = AdvectionSolver(**shared_config, v=v_cw)
    solver_cw.rkorder()

    solver_ccw = AdvectionSolver(**shared_config, v=v_ccw)
    solver_ccw.rkorder()

    cw_solution = solver_cw.u_snapshots[-1][1]
    ccw_solution = np.fliplr(solver_ccw.u_snapshots[-1][1])
    diffs = cw_solution - ccw_solution
    print(f"{l1(diffs)=}")
    print(f"{l2(diffs)=}")
    print(f"{linf(diffs)=}")
    err = l2(diffs)
    assert err < 1e-14
