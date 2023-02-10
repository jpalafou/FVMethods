import pytest
import numpy as np
from util.advection import (
    AdvectionSolver,
    AdvectionSolver_minmod,
    AdvectionSolver_nOrder_MPP,
)
from util.initial_condition import initial_condition1D

# inputs
a = 1  # tranpsort speed
x_bounds = [0, 1]  # spatial domain
T = 2  # solving time


@pytest.mark.parametrize("advection_speed", [1, -1])
def test_periodic_solution_nolimiter(advection_speed):
    """
    one period of avection should match the initial condition
    half a period shouldn't
    do this for a and -a
    """
    h = 0.01
    # x array
    x_interface = np.arange(x_bounds[0], x_bounds[1] + h, h)
    x = 0.5 * (x_interface[:-1] + x_interface[1:])  # x at cell centers
    # time array
    Dt = 0.5 * h / a
    t = np.arange(0, 1 + Dt, Dt)

    # initial conditions (square)
    u0 = initial_condition1D(x, "sinus")

    # solution
    nolimiter_solution = AdvectionSolver(
        u0=u0, t=t, h=h, a=advection_speed, order=4
    )
    nolimiter_solution.rk4()
    nolimiter = nolimiter_solution.u
    assert nolimiter[-1] == pytest.approx(u0, rel=1e-6, abs=1e-3)
    assert nolimiter[int(len(t) / 2)] != pytest.approx(u0, rel=1e-6, abs=1e-3)


def test_conservation_nolimiter():
    """
    Advection solutions should be approximately conservative after 10
    orbits on a fine mesh. the integral of the square wave is 0.5
    """
    h = 0.005
    # x array
    x_interface = np.arange(x_bounds[0], x_bounds[1] + h, h)
    x = 0.5 * (x_interface[:-1] + x_interface[1:])  # x at cell centers
    # time array
    Dt = 0.16 * h / a
    t = np.arange(0, 10 + Dt, Dt)

    # initial conditions (square)
    u0 = np.array(
        [np.heaviside(i - 0.25, 1) - np.heaviside(i - 0.75, 1) for i in x]
    )

    # solution
    minmod_solution = AdvectionSolver_minmod(u0=u0, t=t, h=h, a=a)
    minmod_solution.euler()
    minmod = minmod_solution.u
    assert np.trapz(minmod[-1], x) == pytest.approx(0.5)


def test_conservation_minmod():
    """
    Advection solutions should be approximately conservative after 10
    orbits on a fine mesh. the integral of the square wave is 0.5
    """
    h = 0.005
    # x array
    x_interface = np.arange(x_bounds[0], x_bounds[1] + h, h)
    x = 0.5 * (x_interface[:-1] + x_interface[1:])  # x at cell centers
    # time array
    Dt = 0.16 * h / a
    t = np.arange(0, 10 + Dt, Dt)

    # initial conditions (square)
    u0 = np.array(
        [np.heaviside(i - 0.25, 1) - np.heaviside(i - 0.75, 1) for i in x]
    )

    # solution
    minmod_solution = AdvectionSolver_minmod(u0=u0, t=t, h=h, a=a)
    minmod_solution.euler()
    minmod = minmod_solution.u
    assert np.trapz(minmod[-1], x) == pytest.approx(0.5)


def test_conservation_mpp():
    """
    Advection solutions should be approximately conservative after 10
    orbits on a fine mesh. the integral of the square wave is 0.5
    """
    h = 0.005
    # x array
    x_interface = np.arange(x_bounds[0], x_bounds[1] + h, h)
    x = 0.5 * (x_interface[:-1] + x_interface[1:])  # x at cell centers
    # time array
    Dt = 0.16 * h / a
    t = np.arange(0, 10 + Dt, Dt)

    # initial conditions (square)
    u0 = np.array(
        [np.heaviside(i - 0.25, 1) - np.heaviside(i - 0.75, 1) for i in x]
    )

    # solution
    mpp_solution = AdvectionSolver_nOrder_MPP(u0=u0, t=t, h=h, a=a, order=3)
    mpp_solution.ssp_rk3()
    mpp = mpp_solution.u
    assert np.trapz(mpp[-1], x) == pytest.approx(0.5)


def test_minmod_monotonicity_preservation():
    """
    the minmod solution should be monotonicity preserving
    """
    h = 0.02
    # x array
    x_interface = np.arange(x_bounds[0], x_bounds[1] + h, h)
    x = 0.5 * (x_interface[:-1] + x_interface[1:])  # x at cell centers
    # time array
    Dt = 0.5 * h / a
    t = np.arange(0, T + Dt, Dt)

    # initial conditions (square)
    u0 = np.array(
        [np.heaviside(i - 0.25, 1) - np.heaviside(i - 0.75, 1) for i in x]
    )

    # no limiter solution, should NOT be monotonicity preserving
    nolimiter_solution = AdvectionSolver(u0=u0, t=t, h=h, a=a, order=3)
    nolimiter_solution.rk3()
    nolimiter = nolimiter_solution.u

    # minmod solution, SHOULD be monotonicity preserving
    minmod_solution = AdvectionSolver_minmod(u0=u0, t=t, h=h, a=a)
    minmod_solution.euler()
    minmod = minmod_solution.u

    # monotonicity is tested near the left half of the square
    test_region = [i for i in range(len(x)) if x[i] > 0.05 and x[i] < 0.45]
    assert not all(
        [nolimiter[-1][i + 1] - nolimiter[-1][i] > 0 for i in test_region]
    )
    assert all([minmod[-1][i + 1] - minmod[-1][i] > 0 for i in test_region])


def test_maxmimum_principle_preservation():
    """
    the mpp solution should be maximum principle preserving when the
    Courant condition is satisfied and not be when the condition is
    excessively violated
    """
    h = 0.02
    # x array
    x_interface = np.arange(x_bounds[0], x_bounds[1] + h, h)
    x = 0.5 * (x_interface[:-1] + x_interface[1:])  # x at cell centers

    # initial conditions (square)
    u0 = np.array(
        [np.heaviside(i - 0.25, 1) - np.heaviside(i - 0.75, 1) for i in x]
    )

    # Courant satisfied < 0.05 for 8th order
    C = 0.049
    t = np.arange(0, T + C * h / a, C * h / a)
    mpp_solution = AdvectionSolver_nOrder_MPP(u0=u0, t=t, h=h, a=a, order=8)
    mpp_solution.ssp_rk3()
    mpp = mpp_solution.u

    # Courant violated
    C = 0.8
    t = np.arange(0, T + C * h / a, C * h / a)
    nompp_solution = AdvectionSolver_nOrder_MPP(u0=u0, t=t, h=h, a=a, order=8)
    nompp_solution.ssp_rk3()
    nompp = nompp_solution.u

    assert all([mpp[-1][i] > 0 and mpp[-1][i] < 1 for i in range(len(x))])
    assert not all(
        [nompp[-1][i] > 0 and nompp[-1][i] < 1 for i in range(len(x))]
    )
