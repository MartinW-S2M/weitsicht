import numpy as np

from weitsicht.geometry.coplanar_collinear import is_coplanar
from weitsicht.geometry.interpolation_bilinear import bilinear_interpolation
from weitsicht.geometry.intersection_bilinear import multilinear_poly_intersection


def test_multilinear_poly_intersection():
    points = np.array(
        [
            [434.0, 546.0, 1022.75415039],
            [434.0, 547.0, 1024.71032715],
            [435.0, 546.0, 1022.11627197],
            [435.0, 547.0, 1022.40734863],
        ]
    )

    pos = np.array([433.34261229, 546.22363908, 1041.6013568])
    ray = np.array([0.04988232, 0.03741223, -0.99805415])

    interp = multilinear_poly_intersection(points, pos, ray)

    assert interp is not None
    assert np.allclose(interp, np.array([[434.22161617, 546.88290063, 1024.014094]]), atol=1e-8, rtol=0)

    points = np.array(
        [
            [434.0, 546.0, 1022.75415039],
            [434.0, 547.0, 1024.71032715],
            [435.0, 546.0, 1022.11627197],
            [435.0, 547.0, 1022.40734863],
        ]
    )

    pos = np.array([434.34261229, 546.22363908, 1500])
    ray = np.array([0.00, 0.0, -1])

    interp = multilinear_poly_intersection(points, pos, ray)

    # coplanar
    points = np.array(
        [
            [434.0, 546.0, 1022],
            [434.0, 547.0, 1022],
            [435.0, 546.0, 1022],
            [435.0, 547.0, 1022],
        ]
    )

    pos = np.array([434.34261229, 546.22363908, 1500])
    ray = np.array([0.00, 0.0, -1])

    interp = multilinear_poly_intersection(points, pos, ray)
    assert interp is not None
    assert np.allclose(interp, np.array([434.34261229, 546.22363908, 1022]), atol=1e-9, rtol=0)

    # Double Intersection
    points = np.array(
        [
            [434.0, 546.0, 1000],
            [434.0, 547.0, 1022],
            [435.0, 546.0, 1022],
            [435.0, 547.0, 1000],
        ]
    )

    pos = points[0, :] + np.array([0, 0, 1])
    ray = points[3, :] - points[0, :]

    interp = multilinear_poly_intersection(points, pos, ray)
    assert interp is not None
    assert np.allclose(interp[2], 1001, atol=1e-9, rtol=0)

    # No intersection
    points = np.array(
        [
            [434.0, 546.0, 1000],
            [434.0, 547.0, 1022],
            [435.0, 546.0, 1022],
            [435.0, 547.0, 1000],
        ]
    )

    pos = points[0, :] + np.array([0, 0, 100])
    ray = points[3, :] - points[0, :]

    interp = multilinear_poly_intersection(points, pos, ray)
    assert interp is None

    # Low intersection
    points = np.array(
        [
            [434.0, 546.0, 1000],
            [434.0, 547.0, 1022],
            [435.0, 546.0, 1022],
            [435.0, 547.0, 1000],
        ]
    )

    ray = points[3, :] - points[0, :]
    pos = points[0, :] - ray * 1e-7

    interp = multilinear_poly_intersection(points, pos, ray)
    assert interp is not None
    assert np.allclose(interp, points[0, :], atol=1e-9, rtol=0)

    # intersection coplanar and ray parallel
    points = np.array([[0.0, 0.0, 0], [1.0, 0.0, 0], [0.0, 1.0, 1], [1.0, 1.0, 1]])

    assert is_coplanar(*points)

    ray = points[2, :] - points[0, :]
    pos = points[0, :]

    interp = multilinear_poly_intersection(points, pos, ray)
    assert interp is None


def test_multilinear_poly_intersection_matches_bilinear_interpolation_with_large_coordinates():
    points = np.array(
        [
            [556780.0, 429520.0, 1086.5],
            [556780.0, 429530.0, 1086.1],
            [556790.0, 429520.0, 1085.7],
            [556790.0, 429530.0, 1085.2],
        ]
    )
    pos = np.array([556782.0067585, 429523.04801407, 1600.04592051])
    ray = np.array([2.70689117e-09, 7.04219538e-03, -9.94063672e-01])

    for z_offset in (0.0, 100_000_000.0):
        shifted_points = points.copy()
        shifted_points[:, 2] += z_offset
        shifted_pos = pos.copy()
        shifted_pos[2] += z_offset

        interp = multilinear_poly_intersection(shifted_points, shifted_pos, ray)

        assert interp is not None
        z_bilinear, _normal = bilinear_interpolation(points=shifted_points.tolist(), x=interp[0], y=interp[1])
        assert np.allclose(interp[2], z_bilinear, atol=1e-7, rtol=0)


def test_multilinear_poly_intersection_matches_bilinear_interpolation_for_nearly_linear_ray():
    points = np.array(
        [
            [393.0, 493.0, 1085.64074707],
            [393.0, 494.0, 1085.69702148],
            [394.0, 493.0, 1086.88415527],
            [394.0, 494.0, 1087.1505127],
        ]
    )
    p1 = np.array([393.28004947, 493.30530297, 1100.05846669])
    p2 = np.array([393.24464358, 493.30530298, 1050.0597213])
    ray = (p2 - p1) / np.linalg.norm(p2 - p1)

    interp = multilinear_poly_intersection(points, p=p1, r=ray)

    assert interp is not None
    z_bilinear, _normal = bilinear_interpolation(points=points.tolist(), x=interp[0], y=interp[1])
    assert np.allclose(interp[2], z_bilinear, atol=1e-10, rtol=0)
