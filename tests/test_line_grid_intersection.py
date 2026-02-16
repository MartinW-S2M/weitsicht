from weitsicht.geometry.line_grid_intersection import (
    line_grid_intersection_points,
    raster_index_p1_p2,
)


def test_raster_index_p1_p2():
    res = raster_index_p1_p2([23.23, 300.3], [25.999, 100.0])
    assert res == (23, 26, 100, 301)

    res = raster_index_p1_p2([25.23, 300.3], [23.99999, 100.0])
    assert res == (23, 26, 100, 301)

    res = raster_index_p1_p2([-25.23, 300.3], [23.99999, 100.0])
    assert res == (-26, 24, 100, 301)

    res = raster_index_p1_p2([-25.23, -300.3], [23.99999, -99.2])
    assert res == (-26, 24, -301, -99)

    res = raster_index_p1_p2([-25.23, -300.3], [23.99999, -100.00001])
    assert res == (-26, 24, -301, -100)


def test_line_grid_intersection_points():
    res = line_grid_intersection_points([120.34, 34.33], [120.34, 34.33])

    assert res.shape == (1, 2)

    res = line_grid_intersection_points([120.34, 34.33], [120.34, 34.43])
    assert res.shape == (2, 2)

    res = line_grid_intersection_points([120.34, 34.33], [120.34, 38.43])
    assert res.shape == (6, 2)

    res = line_grid_intersection_points([120.34, 38.43], [120.34, 34.33])
    assert res.shape == (6, 2)

    res = line_grid_intersection_points([120.34, 34.33], [125.34, 34.33])
    assert res.shape == (7, 2)

    res = line_grid_intersection_points([125.34, 34.33], [120.34, 34.33])
    assert res.shape == (7, 2)

    res = line_grid_intersection_points([125.34, 34.33], [120.34, 38.33])
    assert res.shape == (13, 2)
