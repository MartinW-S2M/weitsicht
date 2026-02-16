.. _monoplotting:

============
Monoplotting
============

**Monoplotting** is commonly used as a term for *single-image* mapping where image measurements are converted to 3D
coordinates by intersecting the image viewing ray with a ground model / surface model.

In practical terms:

- pixel -> ray (from camera pose + intrinsics)
- ray + surface (plane / DEM / mesh) -> 3D intersection point

In ``weitsicht`` this workflow is implemented by the image mapping helpers:

- :py:meth:`weitsicht.ImagePerspective.map_points`
- :py:meth:`weitsicht.ImagePerspective.map_center_point`
- :py:meth:`weitsicht.ImagePerspective.map_footprint`

All of them require (directly or indirectly) a mapper that represents the ground/surface model used for the
intersection, for example:

- :doc:`mapper/horizontal_plane`
- :doc:`mapper/raster`
- :doc:`mapper/mesh`
- :doc:`mapper/georef_array`

Related user guide: :doc:`/user_guides/images_projection_mapping`.

Camera pose
-------------------------
The camera pose (exterior orientation) is often described by six parameters: the position of the projection center
(:math:`X_0, Y_0, Z_0`) and three rotation angles (forming a rotation matrix) describing the attitude in space.

In photogrammetry, exterior orientation is often represented by angle triplets such as **OPK** (omega/phi/kappa) or
**AZK/APK** (alpha/zeta/kappa).
See :doc:`/documentation/pose_orientation` for definitions and when each notation is typically used, and
:doc:`/documentation/image/perspective_image` for the basics of perspective imagery and how EOR is used.
