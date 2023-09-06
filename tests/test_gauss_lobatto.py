import numpy as np
import pytest
from finite_volume.mathematiques import gauss_lobatto


def test_1point_quadarture():
    with pytest.raises(BaseException):
        nodes, weights = gauss_lobatto(1)


def test_trivial_quadrature():
    nodes, weights = gauss_lobatto(2)
    assert all(nodes == np.array([-1, 1]))
    assert all(weights == np.array([1, 1]))


def test_3point_quadrature():
    nodes, weights = gauss_lobatto(3)
    assert all(nodes == np.array([-1, 0, 1]))
    assert all(weights == np.array([1 / 3, 4 / 3, 1 / 3]))


def test_4point_quadrature():
    nodes, weights = gauss_lobatto(4)
    assert nodes == pytest.approx(np.array([-1, -np.sqrt(5) / 5, np.sqrt(5) / 5, 1]))
    assert weights == pytest.approx(np.array([1 / 6, 5 / 6, 5 / 6, 1 / 6]))


def test_5point_quadrature():
    nodes, weights = gauss_lobatto(5)
    assert nodes == pytest.approx(
        np.array([-1, -np.sqrt(21) / 7, 0, np.sqrt(21) / 7, 1])
    )
    assert weights == pytest.approx(
        np.array([1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10])
    )
