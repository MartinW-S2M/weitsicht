===========
Type Guards
===========

``weitsicht`` provides a few small helper functions that act as *type guards*.
They are mainly useful for static type checking (pyright, mypy), where they can
help narrow base classes (like ``ImageBase``) to concrete subclasses (like
``ImagePerspective``) based on the runtime type discriminator.

Available guards
================

- ``is_opencv_camera``: narrow ``CameraBasePerspective`` → ``CameraOpenCVPerspective``
- ``is_camera_type``: check a camera discriminator (narrowing works when called with a literal, e.g. ``CameraType.OpenCV``)
- ``is_perspective_image``: narrow ``ImageBase`` → ``ImagePerspective``
- ``is_ortho_image``: narrow ``ImageBase`` → ``ImageOrtho``

They are available from :mod:`weitsicht` (top-level) and from :mod:`weitsicht.type_guards`.

Examples
========

.. code-block:: python

   from weitsicht import ImageBase, is_perspective_image

   def process(img: ImageBase) -> None:
       if is_perspective_image(img):
           # here img is treated as ImagePerspective by type checkers
           _ = img.camera
       else:
           # here img is treated as ImageOrtho | ImageBase (depending on your checker)
           pass

.. code-block:: python

   from weitsicht import CameraBasePerspective, CameraType, is_camera_type

   def process_camera(cam: CameraBasePerspective) -> None:
       if is_camera_type(cam, CameraType.OpenCV):
           # here cam is treated as CameraOpenCVPerspective by type checkers
           _ = cam.camera_matrix
