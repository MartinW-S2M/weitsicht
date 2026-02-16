.. _pose-orientation:

=========================================
Image Pose, Attitude, and Rotation Angles
=========================================

In photogrammetry and computer vision, an image's **pose** (also called *attitude* or *orientation*) describes how the
camera coordinate system is rotated relative to a world / mapping coordinate system.

In ``weitsicht`` an image is geo-referenced when it has:

- a camera model (intrinsics),
- a position (camera center),
- an orientation (:py:class:`weitsicht.Rotation`),
- and a CRS definition for that pose (the *perspective image CRS*).


Rotation matrix (what it means)
===============================

``weitsicht.Rotation`` stores a **3×3 rotation matrix** :math:`R`.

By convention in ``weitsicht``, the matrix maps **camera CRS → world CRS** (the perspective image CRS):

.. math::

    \mathbf{v}_{world} = R \, \mathbf{v}_{cam}

To transform a 3D point :math:`\mathbf{X}` in world coordinates into camera coordinates you typically use:

.. math::

    \mathbf{X}_{cam} = R^{T} (\mathbf{X}_{world} - \mathbf{P}_0)

where :math:`\mathbf{P}_0` is the projection center (camera position) in world CRS.

This convention matches how :py:meth:`weitsicht.ImagePerspective.project` and
:py:meth:`weitsicht.ImagePerspective.pixel_to_ray_vector` use the rotation.

.. note::
   In the ``weitsicht`` camera CRS the optical axis points along :math:`-Z_{CAM}`. With the camera-to-world rotation
   :math:`R`, the viewing direction in world CRS is therefore ``-R[:, 2]``.


.. note::
   Some software packages store a **world-to-camera** rotation and compute camera coordinates like
   :math:`\mathbf{X}_{cam} = R \, (\mathbf{X}_{world} - \mathbf{P}_0)`.
   In ``weitsicht`` the stored matrix maps **camera-to-world** (:math:`\mathbf{v}_{world} = R\,\mathbf{v}_{cam}`),
   because then the rotation matrix directly shows how the camera axes are aligned in the world frame (its columns are
   the camera basis vectors expressed in world coordinates). This is also commonly how INS/IMU systems report attitude.
   This is mainly a personal preference that makes debugging easier; both conventions are equivalent by transposing the rotation matrix.

Spatial similarity transformation (notation mapping)
----------------------------------------------------

Some photogrammetry / geodesy texts describe the relation between a local/sensor frame (lowercase) and a
superior/world frame (uppercase) as a 3D **spatial similarity transformation**:

.. math::

    (\mathbf{p} - \mathbf{p}_0) = s \, R^{T} \, (\mathbf{P} - \mathbf{P}_0)

where:

- :math:`\mathbf{p}` is a point in the local/sensor coordinate system,
- :math:`\mathbf{p}_0` is the point of rotation (origin) in the local/sensor coordinate system,
- :math:`s` is a scale factor,
- :math:`\mathbf{P}` is the same point in the superior/world coordinate system,
- :math:`\mathbf{P}_0` is the point of rotation (origin) in the superior/world coordinate system,
- :math:`R` is the 3×3 rotation matrix.

For camera poses in ``weitsicht`` this is a **rigid** transform, i.e. :math:`s = 1`. The local system is the
**camera CRS** whose origin is the camera projection center, so :math:`\mathbf{p}_0 = \mathbf{0}` and
:math:`\mathbf{P}_0` is the camera center in world/perspective CRS. This yields the standard 3D transform used in
``weitsicht``:

.. math::

    \mathbf{X}_{cam} = R^{T} (\mathbf{X}_{world} - \mathbf{P}_0)

and the inverse:

.. math::

    \mathbf{X}_{world} = \mathbf{P}_0 + R \, \mathbf{X}_{cam}

For how 2D image observations (:math:`x, y`) are linked to this pose via interior orientation
(:math:`x_0, y_0, c` and distortion) and the classic collinearity equations, see :doc:`image/perspective_image`.


Two common photogrammetry angle notations
=========================================

There are many valid angle conventions in the literature. The same symbol names may mean different axis orders or
different sign conventions depending on the software stack. Always validate your convention with a known control point.


OPK (omega, phi, kappa) - typical for aerial / near-nadir imagery
-----------------------------------------------------------------

**OPK** (:math:`\omega, \varphi, \kappa`) is the classic photogrammetry notation for image exterior orientation angles.
It is especially common for **aerial imagery** where the camera looks close to nadir:

- :math:`\omega` (omega): small tilt/roll component,
- :math:`\varphi` (phi): small tilt/pitch component,
- :math:`\kappa` (kappa): rotation around the camera Z axis (often related to heading / image rotation).

Practical intuition:

- When images are close to vertical, :math:`\omega` and :math:`\varphi` are usually small.
- :math:`\kappa` often dominates and is close to the map-direction / heading rotation.

.. note::
   OPK is most intuitive for near-nadir imagery. For near-horizontal views, AZK/APK (or quaternions / rotation matrices)
   are often the better choice.

In ``weitsicht`` you can build this with:

.. code-block:: python

    from weitsicht import Rotation
    rot = Rotation.from_opk_degree(omega=1.2, phi=-0.5, kappa=42.0)

OPK rotation matrix definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In ``weitsicht`` the OPK angles define the camera-to-world rotation matrix as:

.. math::

    R_{\omega,\varphi,\kappa} = R_x(\omega)\,R_y(\varphi)\,R_z(\kappa)

with:

.. math::

    R_x(\omega) =
    \begin{bmatrix}
    1 & 0 & 0 \\
    0 & \cos\omega & -\sin\omega \\
    0 & \sin\omega & \cos\omega
    \end{bmatrix}

.. math::

    R_y(\varphi) =
    \begin{bmatrix}
    \cos\varphi & 0 & \sin\varphi \\
    0 & 1 & 0 \\
    -\sin\varphi & 0 & \cos\varphi
    \end{bmatrix}

.. math::

    R_z(\kappa) =
    \begin{bmatrix}
    \cos\kappa & -\sin\kappa & 0 \\
    \sin\kappa & \cos\kappa & 0 \\
    0 & 0 & 1
    \end{bmatrix}

which evaluates to:

.. math::

    R_{\omega,\varphi,\kappa} =
    \begin{bmatrix}
    \cos\varphi\cos\kappa & -\cos\varphi\sin\kappa & \sin\varphi \\
    \cos\omega\sin\kappa + \sin\omega\sin\varphi\cos\kappa &
    \cos\omega\cos\kappa - \sin\omega\sin\varphi\sin\kappa &
    -\sin\omega\cos\varphi \\
    \sin\omega\sin\kappa - \cos\omega\sin\varphi\cos\kappa &
    \sin\omega\cos\kappa + \cos\omega\sin\varphi\sin\kappa &
    \cos\omega\cos\varphi
    \end{bmatrix}

This is the same OPK definition used by :py:class:`weitsicht.Rotation`.


AZK / APK (alpha, zeta, kappa) - typical for terrestrial / horizontal imagery
------------------------------------------------------------------------------

**AZK** (also written **APK**, using :math:`\alpha, \zeta, \kappa`) is commonly used to describe a camera direction
and roll. In ``weitsicht`` the angles returned by :py:attr:`weitsicht.Rotation.apk` are defined on the camera's +Z axis:

- :math:`\alpha` (alpha): azimuth of the camera +Z axis, in the XY plane,
- :math:`\zeta` (zeta): off-nadir angle (0 deg = nadir, 90 deg = horizontal),
- :math:`\kappa` (kappa): rotation around the camera +Z axis.

.. note::
   In the ``weitsicht`` camera CRS the optical axis is :math:`-Z_{CAM}`. If you prefer AZK/APK angles defined on the
   viewing direction (optical axis), you can convert approximately with:

   - :math:`\alpha_{view} = \alpha + 180^\circ` (wrapped to your preferred range)
   - :math:`\kappa_{view} = -\kappa`

This notation is often convenient for **terrestrial / oblique imagery** where the camera looks close to horizontal
(:math:`\zeta \approx 90^\circ`) and you want an immediate interpretation of:

- where the camera is looking (azimuth + up/down tilt),
- and how the image is rolled.

.. note::
   For :math:`\zeta \approx 0^\circ` (near-nadir) or :math:`\zeta \approx 180^\circ` the azimuth/roll split is not
   unique (gimbal lock). In that case OPK, quaternions, or a rotation matrix are usually the better interchange format.

In ``weitsicht`` you can build this with:

.. code-block:: python

    from weitsicht import Rotation
    rot = Rotation.from_apk_degree(alpha=120.0, zeta=90.0, kappa=0.0)


AZK / APK rotation matrix definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In ``weitsicht`` the AZK/APK angles define the camera-to-world rotation matrix as:

.. math::

    R_{\alpha,\zeta,\kappa} = R_z(\alpha)\,R_y(\zeta)\,R_z(\kappa)

with:

.. math::

    R_z(\alpha) =
    \begin{bmatrix}
    \cos\alpha & -\sin\alpha & 0 \\
    \sin\alpha & \cos\alpha & 0 \\
    0 & 0 & 1
    \end{bmatrix}

.. math::

    R_y(\zeta) =
    \begin{bmatrix}
    \cos\zeta & 0 & \sin\zeta \\
    0 & 1 & 0 \\
    -\sin\zeta & 0 & \cos\zeta
    \end{bmatrix}

.. math::

    R_z(\kappa) =
    \begin{bmatrix}
    \cos\kappa & -\sin\kappa & 0 \\
    \sin\kappa & \cos\kappa & 0 \\
    0 & 0 & 1
    \end{bmatrix}

which evaluates to:

.. math::

    R_{\alpha,\zeta,\kappa} =
    \begin{bmatrix}
    \cos\alpha\cos\zeta\cos\kappa - \sin\alpha\sin\kappa &
    -\cos\alpha\cos\zeta\sin\kappa - \sin\alpha\cos\kappa &
    \cos\alpha\sin\zeta \\
    \sin\alpha\cos\zeta\cos\kappa + \cos\alpha\sin\kappa &
    -\sin\alpha\cos\zeta\sin\kappa + \cos\alpha\cos\kappa &
    \sin\alpha\sin\zeta \\
    -\sin\zeta\cos\kappa &
    \sin\zeta\sin\kappa &
    \cos\zeta
    \end{bmatrix}


Other common orientation representations
========================================

Besides OPK and AZK/APK you will often encounter:

- **Yaw/Pitch/Roll** (also called heading/pitch/roll or roll/pitch/yaw): widely used in navigation and IMU/GNSS.
  (Be careful: different communities use different axis orders and sign conventions.)
- **Pan/Tilt/Roll**: common for gimbals and terrestrial camera rigs (conceptually similar to yaw/pitch/roll).
- **Quaternions**: compact, numerically stable, no gimbal lock; very common in SfM/SLAM and robotics outputs.
- **Axis-angle / rotation vector (Rodrigues)**: used in many vision libraries (e.g. OpenCV) and bundle adjustment outputs.
- **Rotation matrix** directly: the most explicit form, often used in academic outputs and for validation/debugging.

If you frequently exchange data with other tools, it can be safer to store the pose as a rotation matrix or quaternion
and only convert to angles for display/interaction.
