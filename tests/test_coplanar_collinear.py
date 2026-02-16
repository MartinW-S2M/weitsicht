from weitsicht.geometry.coplanar_collinear import is_coplanar


def test_is_coplanar():
    p1 = (0, 0, 0)
    p2 = (0, 100, 0)
    p3 = (100, 0, 100)

    p_test = (100, 100, 100)

    assert is_coplanar(p1, p2, p3, p_test)

    p3 = (100, 0, 100.1)

    assert not is_coplanar(p1, p2, p3, p_test)
