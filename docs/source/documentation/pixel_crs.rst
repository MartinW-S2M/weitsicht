
=======================
Pixel Coordinate System
=======================

``weitsicht`` uses a pixel coordinate system whose origin is at the **upper-left image corner**.
So for example the first pixel center is at ``(0.5, 0.5)``.

All pixel coordinates used and returned by image methods are ordered as ``(x, y)`` where:

- ``x`` is the column direction (to the right),
- ``y`` is the row direction (downwards).


.. figure:: /_static/image_crs.svg
  :align: center
  :alt: Pixel coordinate system

  Pixel coordinate system

.. note::
   Pixels returned by ``ImagePerspective.project`` and ``ImageOrtho.project`` always follow this convention, regardless of
   how the underlying camera model defines its internal pixel origin.

   Example: OpenCV uses the center of the upper-left pixel as ``(0, 0)`` in calibration space. ``weitsicht`` accounts
   for this via the base camera class so users always see a consistent pixel CRS.
