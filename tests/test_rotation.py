import numpy as np
import pytest

from weitsicht import Rotation


@pytest.mark.parametrize(
    ("angles", "expected"),
    [
        ((0, 0, 0), [[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        ((0, 0, 90), [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
        ((90, 0, 0), [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),
        ((0, 90, 0), [[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        ((0, -90, 0), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ],
)
def test_rpy_ned_axis_directions(angles, expected):
    rotation = Rotation.from_rpy_ned_degree(*angles)
    np.testing.assert_allclose(rotation.matrix, expected, atol=1e-12)


@pytest.mark.parametrize("angles", [(23, -17, 61), (-45, 90, 120), (35, -90, -170)])
def test_rpy_ned_combined_rotation(angles):
    # Independent reference: compose elementary rotations by their body axes.
    roll, pitch, yaw = np.deg2rad(angles)
    rx = Rotation.from_opk(roll, 0, 0).matrix
    ry = Rotation.from_opk(0, pitch, 0).matrix
    rz = Rotation.from_opk(0, 0, yaw).matrix
    ned_to_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    camera_to_body = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    rotation = Rotation.from_rpy_ned(roll, pitch, yaw)

    np.testing.assert_allclose(rotation.matrix, ned_to_enu @ rz @ ry @ rx @ camera_to_body, atol=1e-12)
    np.testing.assert_allclose(rotation.matrix, Rotation.from_rpy_ned_degree(*angles).matrix, atol=1e-12)
    np.testing.assert_allclose(rotation.matrix.T @ rotation.matrix, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation.matrix) == pytest.approx(1)


@pytest.mark.parametrize("roll", [0, 23, -75, 180])
def test_rpy_ned_line_of_sight(roll):
    pitch, yaw = np.deg2rad([-32, 61])
    rotation = Rotation.from_rpy_ned_degree(roll, -32, 61)
    expected_los = [np.cos(pitch) * np.sin(yaw), np.cos(pitch) * np.cos(yaw), np.sin(pitch)]
    # Roll changes image axes but never the viewing direction.
    np.testing.assert_allclose(rotation.matrix @ [0, 0, -1], expected_los, atol=1e-12)


@pytest.mark.parametrize("angles", [(23, -17, 61), (-120, 45, -170), (0, 0, 0), (75, 89.999999, -40)])
def test_rpy_ned_properties_round_trip(angles):
    rotation = Rotation.from_rpy_ned_degree(*angles)
    np.testing.assert_allclose(rotation.rpy_ned_degree, angles, atol=1e-9)
    np.testing.assert_allclose(rotation.rpy_ned, np.deg2rad(angles), atol=1e-12)
    target = Rotation(np.eye(3))
    target.rpy_ned = rotation.rpy_ned
    np.testing.assert_allclose(target.matrix, rotation.matrix, atol=1e-12)
    target.rpy_ned_degree = np.array(angles)
    np.testing.assert_allclose(target.matrix, rotation.matrix, atol=1e-12)


@pytest.mark.parametrize("pitch", [-90, 90])
@pytest.mark.parametrize("roll,yaw", [(35, 70), (-120, 170), (0, -45)])
def test_rpy_ned_gimbal_lock_roll_zero(pitch, roll, yaw):
    rotation = Rotation.from_rpy_ned_degree(roll, pitch, yaw)
    expected_yaw = yaw - np.sign(pitch) * roll
    expected_yaw = (expected_yaw + 180) % 360 - 180
    np.testing.assert_allclose(rotation.rpy_ned_degree, [0, pitch, expected_yaw], atol=1e-10)
    np.testing.assert_allclose(Rotation.from_rpy_ned(*rotation.rpy_ned).matrix, rotation.matrix, atol=1e-12)


@pytest.mark.parametrize("pitch", [-90 - 1e-11, -90 + 1e-11, 90 - 1e-11, 90 + 1e-11, 120, -120])
def test_rpy_ned_near_and_beyond_vertical(pitch):
    rotation = Rotation.from_rpy_ned_degree(37, pitch, -53)
    angles = rotation.rpy_ned
    assert np.all(np.isfinite(angles))
    assert -np.pi / 2 <= angles[1] <= np.pi / 2
    np.testing.assert_allclose(Rotation.from_rpy_ned(*angles).matrix, rotation.matrix, atol=1e-12)


def test_rpy_ned_arbitrary_camera_matrix_round_trip():
    # Exercise the inverse on matrices not built by the RPY constructor.
    rng = np.random.default_rng(42)
    for angles in rng.uniform(-180, 180, size=(100, 3)):
        rotation = Rotation.from_opk_degree(*angles)
        np.testing.assert_allclose(Rotation.from_rpy_ned(*rotation.rpy_ned).matrix, rotation.matrix, atol=1e-12)


@pytest.mark.parametrize("attribute", ["rpy_ned", "rpy_ned_degree"])
@pytest.mark.parametrize("angles", [np.zeros(2), np.zeros(4), np.zeros((1, 3)), np.array(0)])
def test_rpy_ned_setter_invalid_shape(attribute, angles):
    rotation = Rotation(np.eye(3))
    with pytest.raises(ValueError, match="Vector3D expected"):
        setattr(rotation, attribute, angles)
    np.testing.assert_array_equal(rotation.matrix, np.eye(3))


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


def test_apk_singular_zeta_zero():
    r = Rotation.from_apk_degree(35, 0, 10)

    assert np.allclose(np.array([0, 0, 45]), r.apk_degree, atol=1e-9, rtol=0)
    assert np.allclose(Rotation.from_apk(*r.apk).matrix, r.matrix, atol=1e-9, rtol=0)


def test_apk_singular_zeta_pi():
    r = Rotation.from_apk_degree(35, 180, 10)

    assert np.allclose(np.array([0, 180, -25]), r.apk_degree, atol=1e-9, rtol=0)
    assert np.allclose(Rotation.from_apk(*r.apk).matrix, r.matrix, atol=1e-9, rtol=0)


def test_apk_singular_zeta_zero_from_matrix():
    r = Rotation(np.diag([-1.0, -1.0, 1.0]))

    assert np.allclose(Rotation.from_apk(*r.apk).matrix, r.matrix, atol=1e-9, rtol=0)


def test_apk_singular_zeta_pi_from_matrix():
    r = Rotation(np.diag([-1.0, 1.0, -1.0]))

    assert np.allclose(Rotation.from_apk(*r.apk).matrix, r.matrix, atol=1e-9, rtol=0)


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
