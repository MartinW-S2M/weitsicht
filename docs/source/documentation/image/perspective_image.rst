.. _perspective-image:

=================
Perspective Image
=================

A perspective image is created by central projection: light rays from 3D points pass through
one projection center and intersect the image plane.

In ``weitsicht``, a perspective image combines:

- a camera model (interior orientation + distortion model),
- an exterior orientation (camera position and rotation),
- an image/pixel coordinate system.

This page describes the full concept, required inputs, and practical implications for mapping.
For practical usage in ``weitsicht``, see :doc:`/user_guides/images_guide`.

Conceptual Split in ``weitsicht``
=================================

For implementation and reasoning, two tightly coupled parts are separated:

- Camera model:
  defines intrinsics and distortion behavior.
- Perspective image:
  combines the image data with georeferencing context (pose, CRS relation, mapping usage).

They are mathematically coupled and must be used together for accurate geo-referenced mapping.

Coordinate Systems Involved
===========================

A perspective image workflow usually involves four coordinate systems:

- 3D point CRS:
  coordinates of terrain/object points, tie points, or reconstructed geometry.
- Perspective image CRS:
  CRS in which image exterior orientation is expressed.
- Camera CRS:
  local frame fixed to the camera body/optical system.
- Pixel/image CRS:
  2D coordinate system of image pixels.

See also:

- :doc:`../camera_crs`
- :doc:`../pixel_crs`

Geo-reference Definition
========================

Within this package, a perspective image is considered geo-referenced when at least:

- a :doc:`camera model <camera>` is available,
- exterior orientation is available,
- the relation to a spatial CRS is defined.

Without these, geometric operations may still run, but results are not reliable for
map-accurate measurements or GIS integration.

Exterior Orientation
====================

Exterior orientation (EO, EOR, XOR, extrinsics) describes camera pose:

- camera center position,
- camera orientation.

Operationally, EOR defines the camera CRS location and orientation in the perspective image CRS.
This convention is critical and must be consistent with axis directions, handedness,
rotation order, and transform direction.

In ``weitsicht``:

- :math:`\mathbf{P}_0` is the camera center position in the *perspective image CRS*.
- :math:`R` is stored as **camera-to-world** rotation (:math:`\mathbf{v}_{world} = R\,\mathbf{v}_{cam}`),
  see :doc:`../pose_orientation`.

Therefore, transforming a 3D point from the perspective image CRS into the camera CRS is done with:

.. math::

    \mathbf{X}_{cam} = R^{T} (\mathbf{X}_{world} - \mathbf{P}_0)

If EOR comes from another software stack, confirm and convert:

- world-to-camera vs camera-to-world representation,
- Euler order and angle sign,
- quaternion convention,
- units (degrees/radians, meters/millimeters).

Interior Orientation (Camera Intrinsics)
========================================

Interior orientation defines the ideal pinhole mapping parameters:

- focal length (or focal lengths in x/y),
- principal point,
- optional skew term.

Real cameras require distortion correction/modeling in addition to ideal intrinsics.
Distortion terms depend on the camera model (radial, tangential, and possibly higher-order terms).
For details on model types, parameter meaning, and distortion validity, see :doc:`camera`.

In ``weitsicht``, camera calibration parameters are defined for a *calibration image size*. When an image is resampled
(e.g. downscaled for speed), pixel coordinates are internally scaled between the current image size and the calibration
size so the same camera model can be reused consistently.

Projection Model
================

A standard perspective model is often written as:

.. math::

    s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
    =
    K \left[ R_{w2c} \mid t_{w2c} \right]
    \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}

where:

- :math:`(X, Y, Z)` is a point expressed in the **perspective image CRS** (or transformed into it),
- :math:`R_{w2c}, t_{w2c}` map perspective/world CRS → camera CRS,
- :math:`K` is the intrinsic matrix,
- :math:`(u, v)` are pixel/image coordinates.

With the ``weitsicht`` convention (:doc:`../pose_orientation`), the corresponding world-to-camera extrinsics are:

.. math::

    R_{w2c} = R^{T}, \qquad t_{w2c} = -R^{T}\mathbf{P}_0

so you may also see:

.. math::

    s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
    =
    K \left[ R^{T} \mid -R^{T}\mathbf{P}_0 \right]
    \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}

Collinearity in vector form (similarity notation)
-------------------------------------------------

In classic photogrammetry, the *collinearity condition* can be expressed in a compact vector form that looks like a
spatial similarity transformation:

.. math::

    (\mathbf{p} - \mathbf{p}_0) = s \, R^{T} \, (\mathbf{P} - \mathbf{P}_0)

Here the **observation** is an image point:

- :math:`\mathbf{p}` is the observed point in the image coordinate system (often embedded as a 3D point on the image plane),
- :math:`\mathbf{p}_0` defines the **interior orientation** (principal point and focal length, e.g. :math:`x_0, y_0, c`),
- :math:`\mathbf{P}` is the observed 3D point in the superior/world system (the perspective image CRS),
- :math:`\mathbf{P}_0` and :math:`R` define the **exterior orientation** (camera center and attitude),
- :math:`s` is an **observation-specific** scale factor (it depends on the point depth; it is not a single global scale).

One common convention is to model the image point on the image plane as:

.. math::

    \mathbf{p} = \begin{bmatrix} x \\ y \\ 0 \end{bmatrix}, \qquad
    \mathbf{p}_0 = \begin{bmatrix} x_0 \\ y_0 \\ c \end{bmatrix}

so that:

.. math::

    \mathbf{p} - \mathbf{p}_0 = \begin{bmatrix} x-x_0 \\ y-y_0 \\ -c \end{bmatrix}

In ``weitsicht``, the interior orientation parameters (principal point, focal length, distortion) live in the camera
model, while :math:`\mathbf{P}_0` and :math:`R` are the pose. The exact sign conventions for :math:`x, y, c` depend on
the chosen image coordinate system and axis directions (see :doc:`../camera_crs` and :doc:`../pixel_crs`).


Collinearity equations (classic photogrammetry form)
----------------------------------------------------

Using the elements :math:`r_{ij}` of the rotation matrix :math:`R` and a projection center
:math:`P_0=(X_0, Y_0, Z_0)`, the collinearity equations are commonly written as:

.. math::

    \frac{x - x_0}{-c} =
    \frac{(X - X_0)\,r_{00} + (Y - Y_0)\,r_{10} + (Z - Z_0)\,r_{20}}
         {(X - X_0)\,r_{02} + (Y - Y_0)\,r_{12} + (Z - Z_0)\,r_{22}}

.. math::

    \frac{y - y_0}{-c} =
    \frac{(X - X_0)\,r_{01} + (Y - Y_0)\,r_{11} + (Z - Z_0)\,r_{21}}
         {(X - X_0)\,r_{02} + (Y - Y_0)\,r_{12} + (Z - Z_0)\,r_{22}}

In this formulation, the camera coordinates are :math:`(x_{cam}, y_{cam}, z_{cam}) = R^{T}\,(X-P_0)` and
:math:`x = x_0 + (-c)\,x_{cam}/z_{cam}`, :math:`y = y_0 + (-c)\,y_{cam}/z_{cam}`.

In practice, distortion correction is applied in addition to this ideal linear projection.

Forward and Inverse Mapping
===========================

Common operations:

- Forward projection:
  3D point -> image pixel.
- Inverse projection (ray casting):
  pixel -> 3D ray from camera center.
- 3D intersection:
  ray + surface model (plane/DEM/mesh) -> 3D point.

This single-image ray/surface intersection step is also often referred to as **Monoplotting**
(see :doc:`/documentation/monoplotting`).

Inverse projection alone does not produce a unique 3D point; an additional geometric constraint
or surface model is required.

When working with undistorted pixels, mask invalid areas first (see the *distortion validity border* on :doc:`camera`).
In code, :py:meth:`weitsicht.ImagePerspective.image_points_inside` can be used to test whether undistorted pixel
coordinates are valid for the current camera model.

In ``weitsicht``, this intersection step is implemented by mapper classes:

- :doc:`../mapper/horizontal_plane` for planar assumptions,
- :doc:`../mapper/raster` for DEM/raster-based surfaces,
- :doc:`../mapper/mesh` for triangle-mesh surfaces,
- :doc:`../mapper/georef_array` for raster based array and full ray-bilinear-patch intersection.

Data Requirements for Reliable Use
==================================

For robust geospatial results, provide:

- calibrated camera model (intrinsics + distortion),
- accurate per-image EOR,
- clearly defined CRS metadata,
- synchronized image and navigation timestamps,
- consistent units and angle conventions.

Optional but often necessary for higher quality:

- lever-arm and boresight calibration,
- ground control/check points,
- a suitable elevation/surface model.

Quality and Accuracy Considerations
===================================

Main error sources:

- wrong CRS or mixed datums,
- wrong rotation convention or axis interpretation,
- poor GNSS/IMU quality or unsynchronized timestamps,
- outdated/inaccurate camera calibration,
- weak geometry (insufficient overlap, no cross-view diversity),
- inappropriate surface model for ray intersection.

Recommended checks:

- reproject known control points and inspect residuals,
- verify that footprints and view directions are physically plausible,
- compare multiple images for consistency over shared ground features.

Practical Checklist
===================

Before processing a dataset, verify:

- camera CRS definition matches your EOR source,
- pixel convention is consistent (:doc:`../pixel_crs`),
- EOR angles and units are converted correctly,
- transform direction is explicit and tested,
- image identifiers and EO records are correctly linked.

Related Pages
=============

- :doc:`../geo-reference_01`
- :doc:`../camera_crs`
- :doc:`../pixel_crs`
- :doc:`../transformation`
- :doc:`/user_guides/images_guide`
