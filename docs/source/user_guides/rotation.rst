=========================
Rotation (Pose Angles)
=========================

``weitsicht.Rotation`` is a small helper around a 3x3 rotation matrix used for image exterior orientation.

It is used for perspective images as the ``orientation=...`` argument of :py:class:`weitsicht.ImagePerspective`.

See also:

- :doc:`/documentation/pose_orientation` (concepts + angle intuitions)
- :doc:`/documentation/camera_crs` (camera axis convention)


Build a rotation from OPK (omega/phi/kappa)
===========================================

OPK is the classic photogrammetry angle notation and is often used for aerial / near-nadir imagery.

.. code-block:: python

    from weitsicht import Rotation

    rot = Rotation.from_opk_degree(omega=1.2, phi=-0.5, kappa=42.0)
    R = rot.matrix

You can also work in radians:

.. code-block:: python

    import numpy as np
    from weitsicht import Rotation

    rot = Rotation.from_opk(omega=np.deg2rad(1.2), phi=np.deg2rad(-0.5), kappa=np.deg2rad(42.0))


Build a rotation from AZK/APK (alpha/zeta/kappa)
================================================

AZK/APK is often convenient for terrestrial / oblique imagery where the camera looks close to horizontal.

.. code-block:: python

    from weitsicht import Rotation

    rot = Rotation.from_apk_degree(alpha=120.0, zeta=90.0, kappa=0.0)

.. note::
   In ``weitsicht`` the camera optical axis points along :math:`-Z_{CAM}`. :py:attr:`weitsicht.Rotation.apk` is defined
   on the camera's +Z axis; see :doc:`/documentation/pose_orientation` for details and conversions.


Get / set angles
================

All angle properties return a 3-element numpy array:

- ``rot.opk`` / ``rot.opk_degree``
- ``rot.apk`` / ``rot.apk_degree``

Example:

.. code-block:: python

    from weitsicht import Rotation
    import numpy as np

    rot = Rotation.from_opk_degree(0.0, 0.0, 0.0)
    rot.opk_degree = np.array([2.0, -1.0, 10.0])
    print(rot.opk_degree)


Use in an image
===============

.. code-block:: python

    import numpy as np
    from pyproj import CRS
    from weitsicht import ImagePerspective, Rotation

    img = ImagePerspective(
        width=4000,
        height=3000,
        camera=camera,
        crs=CRS("EPSG:31256+5778"),
        position=np.array([-1563.025, 338413.602, 500.0]),
        orientation=Rotation.from_opk_degree(omega=0.0, phi=0.1, kappa=90.0),
        mapper=mapper,
    )
