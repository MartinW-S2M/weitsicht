.. _camera-model:

===============
Camera Model
===============

.. note::
   Currently, only the OpenCV pinhole camera model is implemented.
   Contributions adding other camera model families are very welcome, including models for 360-degree imagery.

For practical usage in ``weitsicht``, see :doc:`/user_guides/cameras_distortion`.

The camera model defines how 3D rays are mapped to image pixels and how pixels are mapped back
to rays. In ``weitsicht``, this includes:

- interior orientation (intrinsics),
- lens distortion model,
- model-specific projection and inverse-projection behavior.

See also :doc:`perspective_image` for how the camera model is used together with exterior orientation.

Why the Camera Model Matters
============================

Most geometric errors in image-to-ground mapping are caused by:

- wrong intrinsics,
- wrong distortion parameters,
- wrong interpretation of pixel origin and coordinate convention.

Even small parameter errors can create large errors on the ground, especially at long range and
near image edges.

Intrinsics (Interior Orientation)
=================================

For a central perspective camera, intrinsics are commonly represented by:

.. math::

    K =
    \begin{bmatrix}
    f_x & s   & c_x \\
    0   & f_y & c_y \\
    0   & 0   & 1
    \end{bmatrix}

with:

- :math:`f_x, f_y`: focal lengths in pixel units,
- :math:`(c_x, c_y)`: principal point,
- :math:`s`: skew (often zero).

Typical assumptions:

- pixels are rectangular (``s = 0``),
- :math:`f_x \approx f_y` unless non-square scaling is present.

Distortion
==========

Real lenses deviate from ideal pinhole projection. Common components are:

- radial distortion:
  symmetric displacement from image center,
- tangential distortion:
  decentering effects from lens/sensor misalignment,
- optional higher-order terms:
  model-specific corrections.

Distortion is typically strongest near borders and corners.

Model Families
==============

Typical camera model families used in photogrammetry and vision:

- ideal pinhole:
  no distortion terms, mostly conceptual or pre-undistorted pipelines.
- pinhole + Brown-Conrady style terms:
  common for frame cameras; supports radial and tangential terms.
- fisheye/omnidirectional variants:
  required for wide field-of-view lenses.

Choose the model that matches calibration output. Do not force parameters from one model
into another model family.

Forward and Inverse Usage
=========================

The camera model is used in both directions:

- project:
  3D in camera frame -> pixel.
- unproject:
  pixel -> ray direction in camera frame.

These operations are then combined with exterior orientation and mappers (plane/raster/mesh)
to obtain 3D intersections in geospatial workflows.

Distortion Validity Border
==========================

When undistorting an image or using undistorted pixels, not all undistorted pixels are valid.

Reason:

- each undistorted pixel is mapped through distortion back to a location in the original
  distorted image,
- if that mapped location lies outside the source image extent, that undistorted pixel has no
  valid source sample.

The set of valid undistorted pixels defines a border (often curved) called here the
*distortion validity border*.

.. figure:: /_static/camera_distortion_valid_border.svg
  :align: center
  :alt: Distortion validity border for undistorted pixels

  Undistorted pixels are valid only where their distorted source location remains inside the distorted image.

Depending on distortion strength and how you define the undistorted output (size / ROI / scaling), the mapped border can
also lie *outside* the undistorted image extent. In that case, the full undistorted image is valid (no cropping mask is
needed):

.. figure:: /_static/camera_distortion_valid_border_outside.svg
  :align: center
  :alt: Distortion validity border outside the undistorted image

  When the validity border lies outside the undistorted image, all undistorted pixels map to valid source samples.

Practical implications:

- pixels outside the validity border should be masked as invalid,
- using invalid pixels in ray casting can produce wrong rays/intersections,
- the valid area depends on distortion strength, principal point, and selected undistortion model.

In ``weitsicht``, this is implemented in ``CameraBasePerspective`` (``src/weitsicht/camera/base_perspective.py``):
on initialization, ``__post_init__`` builds a polygonal validity border via ``_generate_distortion_border`` by sampling
the distorted image rectangle and transforming those sampled border points to undistorted space. Later, validity checks
for undistorted image points are done with ``undistorted_image_points_inside``, which tests whether points lie inside
that precomputed border polygon (after scaling image pixels to the camera calibration size). For usage details and
practical examples, see :doc:`/user_guides/cameras_distortion`.

Best Practices
==============

- Keep calibration units and conventions consistent with your implementation.
- Validate reprojection residuals after loading model parameters.
- Document camera model family and parameter order in metadata.
- Do not assume full-image validity after undistortion.
- Combine validity masks with image nodata handling in mapping pipelines.

Related Pages
=============

- :doc:`perspective_image`
- :doc:`../camera_crs`
- :doc:`../pixel_crs`
- :doc:`../mapper/horizontal_plane`
- :doc:`../mapper/raster`
- :doc:`../mapper/mesh`
