Camera Coordinate System
------------------------------------

.. important::
    The camera CRS convention is critical for defining the exterior-orientation rotation correctly.
    See also :doc:`pose_orientation` for how the rotation matrix is defined and used in ``weitsicht``.


The camera coordinate system (camera CRS) is the local 3D reference frame fixed to the camera.
It is required to define how image rays are represented and how exterior orientation is interpreted.

.. figure:: /_static/camera_crs.svg
  :align: center
  :alt: Camera coordinate system

  Camera coordinate system

The camera center :math:`P_{CAM}` is the origin of this system.
Image coordinates and 3D object coordinates are linked through this frame.

Theoretical Background
======================
For a perspective camera, image formation is typically written as:

.. math::

    s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
    =
    K \left[ R \mid t \right]
    \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}

with:

- :math:`K`: camera intrinsics (focal length, principal point, distortion),
- :math:`R, t`: exterior orientation (rotation and translation),
- :math:`(X, Y, Z)`: point in a 3D point CRS,
- :math:`(u, v)`: projected pixel coordinates.

The camera CRS defines the meaning of :math:`(X_c, Y_c, Z_c)` after applying the extrinsics, i.e. before projection to pixels.
Without a strict CRS definition, the same pose values can map to different viewing directions.

Convention in ``weitsicht``
===========================
- Camera origin is at :math:`P_{CAM}`.
- Camera axes :math:`X_{CAM}`, :math:`Y_{CAM}`, :math:`Z_{CAM}` are defined as shown in the figure above
  (:math:`X_{CAM}` right, :math:`Y_{CAM}` up, :math:`Z_{CAM}` backwards; the optical axis points along :math:`-Z_{CAM}`).
- The principal point :math:`(x_0, y_0)` is defined in image coordinates.
- Pixel coordinate details are defined in :doc:`pixel_crs`.

.. note::
    Some software stacks use different sign conventions. For example, OpenCV commonly uses
    :math:`X` right, :math:`Y` down, :math:`Z` forward. ``weitsicht`` converts these conventions inside the camera model
    implementation so the public camera CRS stays consistent with the definition above.

.. important::
    In practice there are often four different coordinate systems:

    - 3D point CRS (coordinates of reconstructed/surveyed points),
    - perspective image CRS (the frame in which image orientation is provided),
    - camera CRS (local camera frame),
    - pixel/image CRS (2D pixel coordinates).

    The image exterior orientation (EOR) describes the camera CRS position and orientation in the perspective image CRS.

    This definition is not universal across computer-vision and photogrammetry tools.
    Therefore, exterior orientation from another software package must be converted before use in ``weitsicht``.

    Always verify these items before importing poses:

    - axis directions and handedness,
    - transform direction (3D-point-CRS to camera vs camera to 3D-point-CRS),
    - rotation convention (Euler order, quaternion convention, matrix layout),
    - unit consistency (meters vs millimeters, radians vs degrees),
    - pixel origin convention (top-left corner vs top-left pixel center; see :doc:`pixel_crs`).

Combined CRS Overview
=====================
The following figure shows how the four systems are linked:
3D point CRS, perspective image CRS, camera CRS, and pixel/image CRS.

.. figure:: /_static/crs_all.svg
  :align: center
  :alt: Combined CRS overview

  Link between 3D point CRS, perspective image CRS, camera CRS, and pixel/image CRS.

References
==========
- OpenCV camera model and projection equations: `OpenCV calib3d <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`_
- OpenCV extrinsics convention in pose estimation: `OpenCV solvePnP <https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html>`_
- Example of a different published camera-frame convention: `ROS Camera Info and coordinate conventions <https://docs.ros.org/en/rolling/p/image_pipeline/camera_info.html>`_
- Example of another convention used in SfM outputs: `COLMAP output format <https://colmap.github.io/format.html>`_
