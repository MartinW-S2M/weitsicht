import numpy as np

from weitsicht.geometry.coplanar_collinear import is_coplanar
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
