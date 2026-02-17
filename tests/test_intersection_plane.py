import numpy as np

from weitsicht.geometry.intersection_plane import intersection_plane_mat_operation


def test_intersection_plane_mat_operation_supports_per_ray_planes():
    line_vec = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    line_point = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    plane_point = np.array([[0.0, 0.0, -10.0], [5.0, 0.0, 0.0]])
    plane_normal = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    intersect, valid = intersection_plane_mat_operation(
        line_vec=line_vec,
        line_point=line_point,
        plane_point=plane_point,
        plane_normal=plane_normal,
    )

    assert np.all(valid)
    assert np.allclose(intersect, np.array([[0.0, 0.0, -10.0], [5.0, 0.0, 0.0]]), atol=1e-12, rtol=0.0)
