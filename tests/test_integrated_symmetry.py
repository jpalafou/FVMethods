import pytest
import numpy as np
import os
from finite_volume.advection import AdvectionSolver
from finite_volume.initial_conditions import generate_ic
from finite_volume.utils import dict_combinations

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
            SED=[True],
        )
    ),
    *dict_combinations(
        dict(
            aposteriori_limiting=[True],
            fallback_limiter=["minmod", "moncen"],
            convex=[False, True],
            hancock=[True],
            fallback_to_first_order=[False, True],
            SED=[True],
        )
    ),
]


@pytest.mark.parametrize("p", range(8))
@pytest.mark.parametrize("a", [1, -1])
@pytest.mark.parametrize("k", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("ic_type__PAD", [("sinus", (-1, 1)), ("square", (0, 1))])
@pytest.mark.parametrize("limiter_config", limiter_configs_1d)
def test_translation_equivariance_1d(p, a, k, ic_type__PAD, limiter_config):
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
    err = linf(diffs)
    assert err < 1e-10


@pytest.mark.parametrize("p", range(8))
@pytest.mark.parametrize("ic_type__PAD", [("sinus", (-1, 1)), ("square", (0, 1))])
@pytest.mark.parametrize("limiter_config", limiter_configs_1d)
def test_velocity_equivariance_1d(p, ic_type__PAD, limiter_config):
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
    err = linf(diffs)
    assert err < 1e-10


"""
2D
"""

limiter_configs_2d = [
    dict(),
    *dict_combinations(
        dict(
            apriori_limiting=[True],
            mpp_lite=[False, True],
            SED=[True],
        )
    ),
    *dict_combinations(
        dict(
            aposteriori_limiting=[True],
            fallback_limiter=["minmod", "moncen"],
            convex=[False, True],
            hancock=[True],
            fallback_to_first_order=[False, True],
            SED=[True],
        )
    ),
    *dict_combinations(
        dict(
            aposteriori_limiting=[True],
            fallback_limiter=["PP2D"],
            convex=[False, True],
            hancock=[True],
            fallback_to_first_order=[False],
            SED=[True],
        )
    ),
]


@pytest.mark.parametrize("p", range(8))
@pytest.mark.parametrize("a", [1, -1])
@pytest.mark.parametrize("k", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("ic_type__PAD", [("sinus", (-1, 1)), ("square", (0, 1))])
@pytest.mark.parametrize("quadrature", ["gauss-legendre", "transverse"])
@pytest.mark.parametrize("limiter_config", limiter_configs_2d)
def test_translation_equivariance_2d(p, a, k, ic_type__PAD, quadrature, limiter_config):
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
    err = linf(diffs)
    assert err < 1e-10


@pytest.mark.parametrize("p", range(8))
@pytest.mark.parametrize("ic_type__PAD", [("sinus", (-1, 1)), ("square", (0, 1))])
@pytest.mark.parametrize("n_rotations", [0, 1, 2, 3])
@pytest.mark.parametrize("quadrature", ["gauss-legendre", "transverse"])
@pytest.mark.parametrize("limiter_config", limiter_configs_1d)
def test_velocity_equivariance_2d(
    p, ic_type__PAD, n_rotations, quadrature, limiter_config
):
    """
    C_n in [C_1, C_2, C_3, C_4]
    advection_solution(u(x, y), (v, 0)) = C_n^-1 advection_...((u(x, y),  C_n (v, 0))
    """
    ic_type, PAD = ic_type__PAD

    def u0(x, y):
        return generate_ic(type=ic_type, x=x, y=y)

    def u0_rotated(x, y):
        return np.rot90(u0(x, y), k=n_rotations, axes=(1, 0))

    u0_rotated.__name__ += f"_{n_rotations}"

    vx_vy = (1, 0)
    vx_vy_C4 = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

    shared_config = dict(
        **limiter_config,
        save_directory=test_directory,
        PAD=PAD,
        n=(64,),
        order=p + 1,
        flux_strategy=quadrature,
        courant=0.8,
        snapshot_dt=1,
    )

    # baseline
    solver = AdvectionSolver(
        **shared_config,
        u0=u0,
        v=vx_vy,
    )
    solver.rkorder()

    # reflected
    solver_rotated = AdvectionSolver(
        **shared_config,
        u0=u0_rotated,
        v=vx_vy_C4[n_rotations],
    )
    solver_rotated.rkorder()

    # check equivariance
    diffs = solver.u_snapshots[-1][1] - np.rot90(
        solver_rotated.u_snapshots[-1][1], k=-n_rotations, axes=(1, 0)
    )
    print(f"{l1(diffs)=}")
    print(f"{l2(diffs)=}")
    print(f"{linf(diffs)=}")
    err = linf(diffs)
    assert err < 1e-10
