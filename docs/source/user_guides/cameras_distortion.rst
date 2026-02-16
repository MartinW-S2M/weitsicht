==============================
Cameras & Distortion Borders
==============================

.. note::
   Currently, only the OpenCV pinhole camera model is implemented.
   Contributions adding other camera model families are very welcome, including models for 360-degree imagery.

Cameras convert between pixel coordinates and camera rays.
``CameraOpenCVPerspective`` implements the OpenCV pinhole model with radial (k1-k4) and tangential (p1-p2) distortion.

For the full reference, see :doc:`/documentation/image/camera`.

Key properties
--------------
- ``param_dict`` / ``from_dict``: serialize and rebuild cameras (used by metadata loaders and scene files).
- ``focal_length_for_gsd_in_pixel``: used to compute ground sampling distance during mapping.
- ``principal_point(image_size)``: returns the pixel center for the current image dimensions.
- ``origin``: shift applied when converting between image pixels and calibration space (OpenCV origin is pixel center, so ``[0.5, 0.5]``).

Distortion border
-----------------
The distortion border is the set of undistorted pixels that can be mapped back to valid distorted pixels
inside the source image extent. Pixels outside this border are invalid for distortion-aware projection.

Implementation details (``src/weitsicht/camera/base_perspective.py``):

- ``CameraBasePerspective.__init__(..., pts_distortion=5)``:
  default sampling density for border construction is ``pts_distortion = 5``.
- ``__post_init__()``:
  automatically builds the border polygon after camera initialization.
- ``_generate_distortion_border(points_between=5)``:
  samples the distorted image rectangle edges and transforms those samples to undistorted space.
- ``distortion_border``:
  stored as a Shapely ``Polygon`` for fast point-in-polygon checks.
- ``undistorted_image_points_inside(...)``:
  tests whether undistorted pixels are within that valid border.

With ``points_between=5``, each side is sampled at 6 positions (endpoint excluded),
resulting in 24 sampled border points in total.

Where this is used in projection:

- In perspective projection, undistorted candidate pixels are first checked with
  ``undistorted_image_points_inside(...)``.
- Only valid pixels are forwarded to distortion mapping.
- Invalid pixels stay masked and should not be used for ray-based mapping.

Practical checks
----------------
.. code-block:: python

    from weitsicht import get_camera_from_dict

    cam = get_camera_from_dict(camera_dict)
    border = cam.distortion_border  # shapely Polygon

    pts = cam.distorted_to_undistorted([[100, 200]], image_size=(4000, 3000))
    inside = cam.undistorted_image_points_inside(pts, image_size=(4000, 3000))

    # optional: map undistorted pixels back to distorted domain
    pts_dist = cam.undistorted_to_distorted(pts, image_size=(4000, 3000))

Guidelines
----------
- Keep ``calib_width`` / ``calib_height`` equal to the image size used for calibration; resampled images should adjust intrinsic values accordingly.
- If distortion terms are unknown, set them to zero; this approximates an undistorted pinhole model and the valid border tends toward the image rectangle.
- If you increase ``pts_distortion``, the border approximation becomes denser (usually more robust near strong edge distortion) with slightly higher initialization cost.
