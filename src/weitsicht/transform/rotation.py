# -----------------------------------------------------------------------
# Copyright 2026 Martin Wieser
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------

"""Rotation matrix utilities."""

from __future__ import annotations

import logging

import numpy as np

from weitsicht.utils import Array3x3, Vector3D

__all__ = ["Rotation"]

logger = logging.getLogger(__name__)


class Rotation:
    """Represent a 3×3 rotation matrix and convert to/from common photogrammetry angle notations.

    R as defined in docu: Vworld = R @ Vcam
    or: Xcam = R.T @ Xworld

    Supported angle sets:

    - **OPK**: omega/phi/kappa (radians/degrees)
    - **APK**: alpha/zeta/kappa (radians/degrees)
    - **RPY NED**: aerospace roll/pitch/yaw converted to camera-to-ENU

    Axis conventions in weitsicht:
    - X axis: left → right in the image
    - Y axis: bottom → top in the image
    - Z axis: backwards (away from the scene)
    - Global system: ENU
    """

    def __init__(self, rotation_matrix: np.ndarray) -> None:
        """Create a :class:`Rotation` from a rotation matrix.

        :param rotation_matrix: Rotation matrix (3×3).
        :type rotation_matrix: numpy.ndarray
        """

        self._rotation_matrix = rotation_matrix

    @classmethod
    def from_opk_degree(cls, omega: float, phi: float, kappa: float) -> Rotation:
        """Create a rotation from omega/phi/kappa angles in degrees.

        :param omega: Omega angle in degrees.
        :type omega: float
        :param phi: Phi angle in degrees.
        :type phi: float
        :param kappa: Kappa angle in degrees.
        :type kappa: float
        :return: Rotation instance.
        :rtype: Rotation
        """

        rotation = cls(np.eye(3, 3, dtype=float))
        rotation.opk_degree = np.array([omega, phi, kappa])
        return rotation

    @classmethod
    def from_opk(cls, omega: float, phi: float, kappa: float) -> Rotation:
        """Create a rotation from omega/phi/kappa angles in radians.

        :param omega: Omega angle in radians.
        :type omega: float
        :param phi: Phi angle in radians.
        :type phi: float
        :param kappa: Kappa angle in radians.
        :type kappa: float
        :return: Rotation instance.
        :rtype: Rotation
        """

        rotation = cls(np.eye(3, 3, dtype=float))
        rotation.opk = np.array([omega, phi, kappa])
        return rotation

    @classmethod
    def from_apk_degree(cls, alpha: float, zeta: float, kappa: float) -> Rotation:
        """Create a rotation from alpha/zeta/kappa angles in degrees.

        ``alpha/zeta/kappa`` (often also written AZK) is a common photogrammetry notation (also used in monoplotting)
        where:

        - ``alpha``: azimuth of the camera +Z axis, relative to +X (East) in the ground CRS,
        - ``zeta``: inclination/tilt relative to nadir (0° = nadir),
        - ``kappa``: rotation around the camera +Z axis.

        :param alpha: Alpha angle in degrees.
        :type alpha: float
        :param zeta: Zeta angle in degrees.
        :type zeta: float
        :param kappa: Kappa angle in degrees.
        :type kappa: float
        :return: Rotation instance.
        :rtype: Rotation
        """

        rotation = cls(np.eye(3, 3, dtype=float))
        rotation.apk_degree = np.array([alpha, zeta, kappa])
        return rotation

    @classmethod
    def from_apk(cls, alpha: float, zeta: float, kappa: float) -> Rotation:
        """Create a rotation from alpha/zeta/kappa angles in radians.

        - ``alpha``: azimuth of the camera +Z axis, relative to +X (East) in the ground CRS,
        - ``zeta``: inclination/tilt relative to nadir (0° = nadir),
        - ``kappa``: rotation around the camera +Z axis.

        :param alpha: Alpha angle in radians.
        :type alpha: float
        :param zeta: Zeta angle in radians.
        :type zeta: float
        :param kappa: Kappa angle in radians.
        :type kappa: float
        :return: Rotation instance.
        :rtype: Rotation
        """

        rotation = cls(np.eye(3, 3, dtype=float))
        rotation.apk = np.array([alpha, zeta, kappa])
        return rotation

    @classmethod
    def from_rpy_ned(cls, roll: float, pitch: float, yaw: float) -> Rotation:
        """Create a camera-to-ENU rotation from NED roll/pitch/yaw in radians.

        Uses the aerospace convention: body axes are forward,
        right, down and ``R_body_to_ned = Rz(yaw) @ Ry(pitch) @ Rx(roll)``
        (intrinsic ZYX). Positive yaw turns North towards East; positive pitch
        raises the nose; positive roll lowers the right side.

        The camera looks along body forward (+X). Camera right is body +Y,
        camera up is body -Z, and camera backwards is body -X. The returned
        matrix therefore maps Weitsicht camera vectors (right, up, backwards)
        into ENU: ``v_enu = rotation.matrix @ v_camera``.
        At zero angles the camera looks North with image top pointing Up.
        The line of sight is ``-rotation.matrix[:, 2]``; pitch -pi/2 looks
        straight down. Roll rotates the image about the line of sight.

        Angles must describe the camera's forward/right/down attitude. Raw INS
        angles require mounting calibration if its axes differ from these.
        Local ENU uses true North; a projected CRS may require an additional
        meridian-convergence correction. No position or CRS is transformed.

        :param roll: Roll about body X, in radians.
        :param pitch: Pitch about body Y, in radians.
        :param yaw: Heading about NED Down, in radians.
        :return: Camera-to-ENU rotation.
        """
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        sy, cy = np.sin(yaw), np.cos(yaw)

        body_to_ned = np.array(
            [
                [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
                [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
                [-sp, sr * cp, cr * cp],
            ]
        )
        ned_to_enu = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        camera_to_body = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        return cls(ned_to_enu @ body_to_ned @ camera_to_body)

    @classmethod
    def from_rpy_ned_degree(cls, roll: float, pitch: float, yaw: float) -> Rotation:
        """Create a camera-to-ENU rotation from NED roll/pitch/yaw in degrees.

        See :meth:`from_rpy_ned` for frame, sequence, and camera mounting conventions.

        :param roll: Roll about body X, in degrees.
        :param pitch: Pitch about body Y, in degrees.
        :param yaw: Heading about NED Down, in degrees.
        :return: Camera-to-ENU rotation.
        """
        return cls.from_rpy_ned(*np.deg2rad([roll, pitch, yaw]))

    @property
    def rpy_ned(self) -> Vector3D:
        """Return camera NED ``[roll, pitch, yaw]`` in radians.

        Inverts :meth:`from_rpy_ned`, including the camera-axis conversion.
        Pitch is in [-pi/2, pi/2]; roll and yaw are in [-pi, pi].
        At gimbal lock (abs(cos(pitch)) <= 1e-12), roll is set to zero
        and the combined rotation is assigned to yaw. The camera matrix is
        preserved to that tolerance, but the original roll/yaw are not unique.
        Near vertical viewing, roll and yaw are numerically ill-conditioned;
        use the matrix for calculations and interpolation.
        """
        ned_to_enu = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        camera_to_body = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        body_to_ned = ned_to_enu.T @ self._rotation_matrix @ camera_to_body.T
        cos_pitch = np.hypot(body_to_ned[0, 0], body_to_ned[1, 0])
        pitch = np.arctan2(-body_to_ned[2, 0], cos_pitch)
        if cos_pitch <= 1e-12:
            roll = 0.0
            yaw = np.arctan2(-body_to_ned[0, 1], body_to_ned[1, 1])
        else:
            roll = np.arctan2(body_to_ned[2, 1], body_to_ned[2, 2])
            yaw = np.arctan2(body_to_ned[1, 0], body_to_ned[0, 0])
        return np.array([roll, pitch, yaw])

    @rpy_ned.setter
    def rpy_ned(self, angles: Vector3D):
        """Set camera NED ``[roll, pitch, yaw]`` in radians (shape ``(3,)``)."""
        angles = np.asarray(angles, dtype=float)
        if angles.shape != (3,):
            raise ValueError(f"Vector3D expected, shape is: {angles.shape}")
        self._rotation_matrix = type(self).from_rpy_ned(*angles).matrix

    @property
    def rpy_ned_degree(self) -> Vector3D:
        """Return camera NED ``[roll, pitch, yaw]`` in degrees.

        See :attr:`rpy_ned` for ranges and the roll=0 gimbal-lock convention.
        """
        return np.rad2deg(self.rpy_ned)

    @rpy_ned_degree.setter
    def rpy_ned_degree(self, angles: Vector3D):
        """Set camera NED ``[roll, pitch, yaw]`` in degrees (shape ``(3,)``)."""
        self.rpy_ned = np.deg2rad(angles)

    @property
    def matrix(self) -> Array3x3:
        """Return the rotation matrix.

        :return: Rotation matrix (3×3).
        :rtype: Array3x3
        """
        return self._rotation_matrix

    @matrix.setter
    def matrix(self, rot: Array3x3):
        """Set the rotation matrix.

        :param rot: Rotation matrix (3×3).
        :type rot: Array3x3
        """
        self._rotation_matrix = rot

    @property
    def opk(self) -> Vector3D:
        """Convert the rotation matrix to omega/phi/kappa angles in radians.

        :return: ``[omega, phi, kappa]`` in radians.
        :rtype: Vector3D
        """

        omega_rad = np.arctan2(-self._rotation_matrix[1, 2], self._rotation_matrix[2, 2])

        # avoid a np domain error: argument of arcsin must be within [-1,1]
        phi_rad = np.arcsin(max(-1.0, min(1.0, self._rotation_matrix[0, 2])))
        kappa_rad = np.arctan2(-self._rotation_matrix[0, 1], self._rotation_matrix[0, 0])
        op_vec_rad = np.array([omega_rad, phi_rad, kappa_rad])

        return op_vec_rad

    @opk.setter
    def opk(self, opk_vec_rad: Vector3D):
        """Convert omega/phi/kappa angles (radians) to a rotation matrix.

        :param opk_vec_rad: ``[omega, phi, kappa]`` in radians.
        :type opk_vec_rad: Vector3D
        :raises ValueError: If the input vector does not have shape ``(3,)``.
        """
        if opk_vec_rad.shape == (3,):  # convert to rotation matrix
            omega = opk_vec_rad[0]
            phi = opk_vec_rad[1]
            kappa = opk_vec_rad[2]

            sin_omega = np.sin(omega)
            cos_omega = np.cos(omega)
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            sin_kappa = np.sin(kappa)
            cos_kappa = np.cos(kappa)

            # This is R as defined in docu: Vworld = R @ Vcam
            # or: Xcam = R.T @ Xworld
            opk_rot_mat = np.array(
                [
                    [cos_phi * cos_kappa, -cos_phi * sin_kappa, sin_phi],
                    [
                        cos_omega * sin_kappa + sin_omega * sin_phi * cos_kappa,
                        cos_omega * cos_kappa - sin_omega * sin_phi * sin_kappa,
                        -sin_omega * cos_phi,
                    ],
                    [
                        sin_omega * sin_kappa - cos_omega * sin_phi * cos_kappa,
                        sin_omega * cos_kappa + cos_omega * sin_phi * sin_kappa,
                        cos_omega * cos_phi,
                    ],
                ]
            )

            self._rotation_matrix = opk_rot_mat

        else:
            raise ValueError(f"Vector3D expected, shape is: {opk_vec_rad.shape}")

    @property
    def opk_degree(self) -> Vector3D:
        """Return omega/phi/kappa angles in degrees.

        :return: ``[omega, phi, kappa]`` in degrees.
        :rtype: Vector3D
        """

        return self.opk / np.pi * 180

    @opk_degree.setter
    def opk_degree(self, opk_vec_degree: Vector3D):
        """Convert omega/phi/kappa angles (degrees) to a rotation matrix.

        :param opk_vec_degree: ``[omega, phi, kappa]`` in degrees.
        :type opk_vec_degree: Vector3D
        :raises ValueError: If the input vector does not have shape ``(3,)``.
        """
        if opk_vec_degree.shape == (3,):
            self.opk = opk_vec_degree * np.pi / 180
        else:
            raise ValueError(f"Vector3D expected, shape is: {opk_vec_degree.shape}")

    @property
    def apk(self) -> Vector3D:
        """Convert the rotation matrix to alpha/zeta/kappa angles in radians.

        In ``weitsicht`` the camera CRS uses a backwards Z axis. Therefore this property uses a ZYZ Euler
        decomposition of the camera's **+Z axis**:

        - ``alpha``: azimuth of the camera +Z axis (XY plane),
        - ``zeta``: off-nadir angle (0 = nadir, 90° = horizontal),
        - ``kappa``: rotation around the camera +Z axis.

        :return: ``[alpha, zeta, kappa]`` in radians.
        :rtype: Vector3D
        """

        # avoid a np domain error: argument of arccos must be within [-1,1]
        cos_zeta = max(-1.0, min(1.0, float(self._rotation_matrix[2, 2])))
        zeta_rad = float(np.arccos(cos_zeta))

        sin_zeta = float(np.sin(zeta_rad))
        if abs(sin_zeta) < 1e-12:
            # Gimbal lock: alpha and kappa are not uniquely defined for zeta ~ 0 or pi.
            # Convention here: set alpha=0 and fold everything into kappa.
            alpha_internal = 0.0
            if cos_zeta > 0:
                kappa_internal = float(np.arctan2(self._rotation_matrix[1, 0], self._rotation_matrix[0, 0]))
            else:
                kappa_internal = float(np.arctan2(self._rotation_matrix[1, 0], -self._rotation_matrix[0, 0]))
        else:
            alpha_internal = float(np.arctan2(self._rotation_matrix[1, 2], self._rotation_matrix[0, 2]))
            kappa_internal = float(np.arctan2(self._rotation_matrix[2, 1], -self._rotation_matrix[2, 0]))

        return np.array([alpha_internal, zeta_rad, kappa_internal])

    @apk.setter
    def apk(self, apk_vec_rad: Vector3D):
        """Convert alpha/zeta/kappa angles (radians) to a rotation matrix.

        :param apk_vec_rad: ``[alpha, zeta, kappa]`` in radians.
        :type apk_vec_rad: Vector3D
        :raises ValueError: If the input vector does not have shape ``(3,)``.
        """
        if apk_vec_rad.shape == (3,):
            alpha = float(apk_vec_rad[0])
            zeta = float(apk_vec_rad[1])
            kappa = float(apk_vec_rad[2])

            sin_alpha = np.sin(alpha)
            cos_alpha = np.cos(alpha)
            sin_zeta = np.sin(zeta)
            cos_zeta = np.cos(zeta)
            sin_kappa = np.sin(kappa)
            cos_kappa = np.cos(kappa)

            # This is R as defined in docu: Vworld = R @ Vcam
            # or: Xcam = R.T @ Xworld
            apk_rot_mat = np.array(
                [
                    [
                        cos_alpha * cos_zeta * cos_kappa - sin_alpha * sin_kappa,
                        -cos_alpha * cos_zeta * sin_kappa - sin_alpha * cos_kappa,
                        cos_alpha * sin_zeta,
                    ],
                    [
                        sin_alpha * cos_zeta * cos_kappa + cos_alpha * sin_kappa,
                        -sin_alpha * cos_zeta * sin_kappa + cos_alpha * cos_kappa,
                        sin_alpha * sin_zeta,
                    ],
                    [-sin_zeta * cos_kappa, sin_zeta * sin_kappa, cos_zeta],
                ]
            )

            self._rotation_matrix = apk_rot_mat
        else:
            raise ValueError(f"Vector3D expected, shape is: {apk_vec_rad.shape}")

    @property
    def apk_degree(self) -> Vector3D:
        """Return alpha/zeta/kappa angles in degrees.

        :return: ``[alpha, zeta, kappa]`` in degrees.
        :rtype: Vector3D
        """

        return self.apk / np.pi * 180

    @apk_degree.setter
    def apk_degree(self, apk_vec_degree: Vector3D):
        """Convert alpha/zeta/kappa angles (degrees) to a rotation matrix.

        :param apk_vec_degree: ``[alpha, zeta, kappa]`` in degrees.
        :type apk_vec_degree: Vector3D
        :raises ValueError: If the input vector does not have shape ``(3,)``.
        """
        if apk_vec_degree.shape == (3,):
            self.apk = apk_vec_degree * np.pi / 180
        else:
            raise ValueError(f"Vector3D expected, shape is: {apk_vec_degree.shape}")

    def transform_to_crs(self):
        """Transform the rotation into another CRS.

        This is currently not implemented.
        """

        pass
