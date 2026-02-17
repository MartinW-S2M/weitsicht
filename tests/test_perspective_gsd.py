import numpy as np

from weitsicht.camera.opencv_perspective import CameraOpenCVPerspective
from weitsicht.image.perspective import ImagePerspective
from weitsicht.transform.rotation import Rotation


def test_perspective_gsd_uses_normals_tangent_plane():
    camera = CameraOpenCVPerspective(width=5, height=5, fx=2.0, fy=2.0)
    image = ImagePerspective(
        width=5,
        height=5,
        camera=camera,
        position=np.array([0.0, 0.0, 0.0]),
        orientation=Rotation(np.eye(3, dtype=float)),
    )

    pixel = np.array([[3.5, 2.5]], dtype=float)  # 1 px right of principal point in image CRS
    ray = image.pixel_to_ray_vector(pixel)

    # Synthetic "mapping": intersect with horizontal plane z=-10.
    t = (-10.0) / ray[:, 2]
    coordinates = ray * t[:, None]
    mask = np.array([True])

    normals_ok = np.array([[0.0, 0.0, 1.0]])
    gsd_ok, gsd_per_point_ok = image._estimate_gsd_for_mapped_points(
        pixels=pixel,
        ray_vectors=ray,
        coordinates=coordinates,
        normals=normals_ok,
        mask=mask,
        is_undistorted=False,
    )

    assert gsd_ok is not None
    assert gsd_per_point_ok.shape == (1,)
    assert np.isfinite(gsd_per_point_ok[0])
    assert np.isclose(gsd_ok, float(gsd_per_point_ok[0]), atol=1e-12, rtol=0.0)

    # For fx=fy=2 and a plane at z=-10, the tangent-plane spacing is 10/f = 5 units per pixel.
    assert np.isclose(gsd_ok, 5.0, atol=1e-9, rtol=0.0)

    # Without valid normals the method falls back to the range-sphere chord estimate (should differ here).
    normals_nan = np.full((1, 3), np.nan, dtype=float)
    gsd_no_normals, _ = image._estimate_gsd_for_mapped_points(
        pixels=pixel,
        ray_vectors=ray,
        coordinates=coordinates,
        normals=normals_nan,
        mask=mask,
        is_undistorted=False,
    )
    assert gsd_no_normals is not None
    assert gsd_no_normals < gsd_ok
