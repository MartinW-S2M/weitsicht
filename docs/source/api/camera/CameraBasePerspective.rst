
:py:class:`CameraBasePerspective`
=================================

.. currentmodule:: weitsicht

.. rubric::   Main API Methods


.. autosummary::

  CameraBasePerspective.pixel_image_to_camera_crs
  CameraBasePerspective.pts_camara_crs_to_image_pixel
  CameraBasePerspective.undistorted_to_distorted
  CameraBasePerspective.distorted_to_undistorted

.. rubric:: additional helpful Methods

.. autosummary::

  CameraBasePerspective.from_dict
  CameraBasePerspective.undistorted_image_points_inside
  CameraBasePerspective.principal_point
  CameraBasePerspective._generate_distortion_border

.. rubric:: Properties

.. autosummary::

  CameraOpenCVPerspective.param_dict
  CameraOpenCVPerspective.type
  CameraOpenCVPerspective.focal_length_for_gsd_in_pixel
  CameraOpenCVPerspective.origin

.. autoclass:: CameraBasePerspective
   :special-members: __init__
