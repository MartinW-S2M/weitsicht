import numpy as np
import pytest

from weitsicht import Rotation


def test_from_opk_degree():

    r = Rotation.from_opk_degree(45, 0, 45)
    assert np.allclose(np.array([45, 0, 45]), r.opk_degree, atol=1e-9, rtol=0)
    assert np.allclose(np.array([np.pi / 4, 0, np.pi / 4]), r.opk, atol=1e-9, rtol=0)


def test_from_opk():
    r = Rotation.from_opk(np.pi / 4, 0, np.pi / 4)
    assert np.allclose(np.array([45, 0, 45]), r.opk_degree, atol=1e-9, rtol=0)
    assert np.allclose(np.array([np.pi / 4, 0, np.pi / 4]), r.opk, atol=1e-9, rtol=0)


def test_from_apk_degree():
    r = Rotation.from_apk_degree(30, 20, -10)
    assert np.allclose(np.array([30, 20, -10]), r.apk_degree, atol=1e-9, rtol=0)
    assert np.allclose(np.array([np.deg2rad(30), np.deg2rad(20), np.deg2rad(-10)]), r.apk, atol=1e-9, rtol=0)


def test_from_apk():
    r = Rotation.from_apk(np.deg2rad(30), np.deg2rad(20), np.deg2rad(-10))
    assert np.allclose(np.array([30, 20, -10]), r.apk_degree, atol=1e-9, rtol=0)
    assert np.allclose(np.array([np.deg2rad(30), np.deg2rad(20), np.deg2rad(-10)]), r.apk, atol=1e-9, rtol=0)


def test_matrix():
    mat_test = np.array(
        [
            [0.70710678, -0.70710678, 0.0],
            [0.5, 0.5, -0.70710678],
            [0.5, 0.5, 0.70710678],
        ]
    )

    r = Rotation.from_opk_degree(45, 0, 45)
    assert np.allclose(r.matrix, mat_test, atol=1e-6, rtol=0)

    r.matrix = mat_test

    assert np.allclose(np.array([45, 0, 45]), r.opk_degree, atol=1e-9, rtol=0)


def test_opk_degree():

    r = Rotation.from_opk_degree(0, 0, 0)

    assert np.allclose(np.eye(3, 3), r.matrix)

    with pytest.raises(ValueError):
        r.opk_degree = np.array([0, 0, 0, 0])

    assert np.allclose(np.array([0, 0, 0]), r.opk, atol=1e-9, rtol=0)

    r.opk_degree = np.array([45, 0, 45])
    assert np.allclose(np.array([45, 0, 45]), r.opk_degree, atol=1e-9, rtol=0)
    assert np.allclose(np.array([np.pi / 4, 0, np.pi / 4]), r.opk, atol=1e-9, rtol=0)


def test_opk():

    r = Rotation.from_opk_degree(0, 0, 0)
    r.opk = np.array([0, 0, 0])

    assert np.allclose(np.eye(3, 3), r.matrix, atol=1e-9, rtol=0)

    with pytest.raises(ValueError):
        r.opk = np.array([0, 0, 0, 0])

    assert np.allclose(np.array([0, 0, 0]), r.opk, atol=1e-9, rtol=0)


def test_apk_degree():
    r = Rotation.from_apk_degree(0, 0, 0)

    assert np.allclose(np.eye(3, 3), r.matrix)

    with pytest.raises(ValueError):
        r.apk_degree = np.array([0, 0, 0, 0])

    assert np.allclose(np.array([0, 0, 0]), r.apk, atol=1e-9, rtol=0)

    r.apk_degree = np.array([30, 20, -10])
    assert np.allclose(np.array([30, 20, -10]), r.apk_degree, atol=1e-9, rtol=0)
    assert np.allclose(np.array([np.deg2rad(30), np.deg2rad(20), np.deg2rad(-10)]), r.apk, atol=1e-9, rtol=0)


def test_apk():
    r = Rotation.from_apk_degree(0, 0, 0)
    r.apk = np.array([0, 0, 0])

    assert np.allclose(np.eye(3, 3), r.matrix, atol=1e-9, rtol=0)

    with pytest.raises(ValueError):
        r.apk = np.array([0, 0, 0, 0])

    assert np.allclose(np.array([0, 0, 0]), r.apk, atol=1e-9, rtol=0)


@pytest.mark.xfail(reason="Not implemented yet")
def test_transform_to_crs():

    assert True
