import pytest

from weitsicht.geometry.interpolation_bilinear import bilinear_interpolation


def test_bilinear_interpolation():

    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 551847.5 429158.5652
    # Report:
    #   Location: (0.549782479727583P,430.567498733493L)
    #   Band 1:
    #     Value: 952.275286000538

    row = 430.567498733493
    col = 0.549782479727583

    points = [
        [430.5, 0.5, 951.97974],
        [431.5, 0.5, 953.73517],
        [430.5, 1.5, 955.56775],
        [431.5, 1.5, 956.85895],
    ]
    z, _ = bilinear_interpolation(points=points, x=row, y=col)
    assert abs(z - 952.275286000538) < 1e-5

    # another order
    points = [
        [431.5, 1.5, 956.85895],
        [430.5, 0.5, 951.97974],
        [431.5, 0.5, 953.73517],
        [430.5, 1.5, 955.56775],
    ]

    z, _ = bilinear_interpolation(points=points, x=row, y=col)
    assert abs(z - 952.275286000538) < 1e-5

    # x outside of points
    with pytest.raises(ValueError):
        bilinear_interpolation(points=points, x=row + 1, y=col)

    # none rectangle points
    points = [
        [431.5, 1.7, 956.85895],
        [430.5, 0.5, 951.97974],
        [431.5, 0.5, 953.73517],
        [430.5, 1.5, 955.56775],
    ]

    with pytest.raises(ValueError):
        bilinear_interpolation(points=points, x=row, y=col)
