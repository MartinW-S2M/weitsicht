"""WGS84 local tangent plane (ENU/NED) helpers.

This module provides :class:`WGS84LocalTangent`:

- Tangent plane orientation (ECEF↔NED/ENU) derived from lon/lat.
- Optional/lazy computation of the ECEF origin when only rotations are needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pyproj import CRS

from weitsicht.exceptions import CoordinateTransformationError
from weitsicht.transform.coordinates_transformer import CoordinateTransformer
from weitsicht.utils import Array3x3, Vector3D

LocalTangentFrame = Literal["ENU", "NED"]

_CRS_WGS84_GEODETIC_3D = CRS.from_epsg(4979)
_CRS_WGS84_ECEF = CRS.from_epsg(4978)
_CRS_WGS84_ORTHOMETRIC = CRS("EPSG:4326+3855")


def _axis_count(crs: CRS) -> int:
    return len(crs.axis_info)


def _ensure_crs_3d(crs: CRS) -> CRS:
    if _axis_count(crs) >= 3:
        return crs
    return crs.to_3d()


def _ecef_to_ned_matrix(lon_deg: float, lat_deg: float) -> Array3x3:
    """Rotation matrix mapping ECEF vectors into NED vectors at (lon, lat)."""

    lon = float(np.deg2rad(lon_deg))
    lat = float(np.deg2rad(lat_deg))

    sin_lat = float(np.sin(lat))
    cos_lat = float(np.cos(lat))
    sin_lon = float(np.sin(lon))
    cos_lon = float(np.cos(lon))

    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=float)
    east = np.array([-sin_lon, cos_lon, 0.0], dtype=float)
    down = np.array([-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat], dtype=float)

    r_ecef_from_ned = np.column_stack((north, east, down))
    return r_ecef_from_ned.T


@dataclass
class WGS84LocalTangent:
    """Local tangent plane frame tied to a WGS84 origin.

    The tangent plane orientation is defined by lon/lat and stored as ``r_ecef_to_ned``.
    ``origin_ecef`` is optional and can be computed lazily from ``origin_ell`` and ``crs_wgs_84``
    when needed (e.g. for point conversions).
    """

    r_ecef_to_ned: Array3x3
    origin_ell: Vector3D
    crs_wgs_84: CRS
    _origin_ecef: Vector3D | None = None
    crs_ecef: CRS = _CRS_WGS84_ECEF

    @property
    def origin_ecef(self):
        return self._origin_ecef_required()

    @staticmethod
    def _ell_to_ecef(origin_ell: Vector3D, crs_wgs_84: CRS) -> Vector3D:
        origin = np.asarray(origin_ell, dtype=float).reshape(3)
        crs_source = _ensure_crs_3d(CRS.from_user_input(crs_wgs_84))

        try:
            transformer = CoordinateTransformer.from_crs(crs_source, _CRS_WGS84_ECEF)
            if transformer is None:
                return origin.astype(float)
            return transformer.transform(origin)[0, :].astype(float)
        except (CoordinateTransformationError, ValueError) as err:
            raise ValueError(f"Failed to transform {crs_source.to_string()} -> EPSG:4978 for origin_ecef: {err}") from err

    @property
    def r_ned_to_ecef(self) -> Array3x3:
        """Rotation matrix that maps NED vectors to ECEF vectors."""

        return self.r_ecef_to_ned.T

    @property
    def r_ecef_to_enu(self) -> Array3x3:
        """Rotation matrix that maps ECEF vectors to ENU vectors."""

        # v_enu = [E, N, U] = [E, N, -D] for v_ned=[N, E, D]
        s_ned_to_enu = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)
        return (s_ned_to_enu @ self.r_ecef_to_ned).astype(float)

    @property
    def r_enu_to_ecef(self) -> Array3x3:
        """Rotation matrix that maps ENU vectors to ECEF vectors."""

        return self.r_ecef_to_enu.T

    def ell_to_ecef(self, origin_ell: Vector3D | None = None, crs_wgs_84: CRS | None = None) -> Vector3D:
        """Compute and cache the ECEF origin from (lon, lat, h) coordinates.

        This is useful if the frame was created with ``skip_ecef=True``.
        """

        if self._origin_ecef is not None and origin_ell is None and crs_wgs_84 is None:
            return self._origin_ecef

        origin = self.origin_ell if origin_ell is None else origin_ell
        crs_source = self.crs_wgs_84 if crs_wgs_84 is None else CRS.from_user_input(crs_wgs_84)
        xyz = self._ell_to_ecef(origin, crs_source)

        self._origin_ecef = xyz
        return xyz

    def _origin_ecef_required(self) -> Vector3D:
        return self._origin_ecef if self._origin_ecef is not None else self.ell_to_ecef()

    @classmethod
    def from_wgs84ell_crs(
        cls,
        crs_s: CRS | str | int,
        lon_deg: float,
        lat_deg: float,
        h_m: float,
        *,
        skip_ecef: bool = False,
    ) -> WGS84LocalTangent:
        """Create a tangent frame from WGS84 lon/lat and a specified vertical reference CRS."""

        crs_source = _ensure_crs_3d(CRS.from_user_input(crs_s))
        origin_ell = np.array([float(lon_deg), float(lat_deg), float(h_m)], dtype=float)

        origin_ecef: Vector3D | None = None
        if skip_ecef is False:
            origin_ecef = cls._ell_to_ecef(origin_ell, crs_source)

        r_ecef_to_ned = _ecef_to_ned_matrix(lon_deg=float(lon_deg), lat_deg=float(lat_deg))
        return cls(
            r_ecef_to_ned=r_ecef_to_ned,
            origin_ell=origin_ell,
            crs_wgs_84=crs_source,
            _origin_ecef=origin_ecef,
            crs_ecef=_CRS_WGS84_ECEF,
        )

    @classmethod
    def from_wgs84ell_elipsoid(
        cls, lon_deg: float, lat_deg: float, h_m: float, *, skip_ecef: bool = False
    ) -> WGS84LocalTangent:
        """Create a tangent frame from WGS84 lon/lat and ellipsoidal height (EPSG:4979)."""

        return cls.from_wgs84ell_crs(_CRS_WGS84_GEODETIC_3D, lon_deg, lat_deg, h_m, skip_ecef=skip_ecef)

    @classmethod
    def from_wgs84ell_orthometric(
        cls, lon_deg: float, lat_deg: float, h_m: float, *, skip_ecef: bool = False
    ) -> WGS84LocalTangent:
        """Create a tangent frame from WGS84 lon/lat and orthometric height (EPSG:4326+3855)."""

        return cls.from_wgs84ell_crs(_CRS_WGS84_ORTHOMETRIC, lon_deg, lat_deg, h_m, skip_ecef=skip_ecef)

    @classmethod
    def from_crs(cls, x: float, y: float, z: float, crs_s: CRS | str | int) -> WGS84LocalTangent:
        """Create a tangent frame from coordinates in an arbitrary CRS."""

        crs_source = _ensure_crs_3d(CRS.from_user_input(crs_s))

        try:
            transformer = CoordinateTransformer.from_crs(crs_source, _CRS_WGS84_ECEF)
            if transformer is None:
                origin_ecef = np.array([float(x), float(y), float(z)], dtype=float)
            else:
                origin_ecef = transformer.transform(np.array([x, y, z], dtype=float))[0, :].astype(float)
        except (CoordinateTransformationError, ValueError) as err:
            raise ValueError(
                f"Failed to transform {crs_source.to_string()} -> EPSG:4978 for local tangent origin: {err}"
            ) from err

        try:
            transformer_ecef_to_wgs84 = CoordinateTransformer.from_crs(_CRS_WGS84_ECEF, _CRS_WGS84_GEODETIC_3D)
            if transformer_ecef_to_wgs84 is None:  # pragma: no cover
                lon_deg, lat_deg, h_m = origin_ecef[0], origin_ecef[1], origin_ecef[2]
            else:
                coo_ell = transformer_ecef_to_wgs84.transform(origin_ecef)
                lon_deg = float(coo_ell[0, 0])
                lat_deg = float(coo_ell[0, 1])
                h_m = float(coo_ell[0, 2])
        except (CoordinateTransformationError, ValueError) as err:
            raise ValueError(f"Failed to transform EPSG:4978 -> EPSG:4979 to derive lon/lat: {err}") from err

        origin_ell = np.array([float(lon_deg), float(lat_deg), float(h_m)], dtype=float)
        r_ecef_to_ned = _ecef_to_ned_matrix(lon_deg=float(lon_deg), lat_deg=float(lat_deg))
        return cls(
            r_ecef_to_ned=r_ecef_to_ned,
            origin_ell=origin_ell,
            crs_wgs_84=_CRS_WGS84_GEODETIC_3D,
            _origin_ecef=origin_ecef,
            crs_ecef=_CRS_WGS84_ECEF,
        )

    def vector_to_ecef(self, v_local: Vector3D, *, local_frame: LocalTangentFrame) -> Vector3D:
        """Convert a vector in a local tangent frame to ECEF."""

        v = np.asarray(v_local, dtype=float).reshape(3)
        frame = local_frame.upper()
        if frame == "NED":
            return (self.r_ned_to_ecef @ v).astype(float)
        if frame == "ENU":
            return (self.r_enu_to_ecef @ v).astype(float)
        raise ValueError(f"Unsupported local_frame: {local_frame!r}")

    def vector_from_ecef(self, v_ecef: Vector3D, *, local_frame: LocalTangentFrame) -> Vector3D:
        """Convert an ECEF vector to a local tangent frame."""

        v = np.asarray(v_ecef, dtype=float).reshape(3)
        frame = local_frame.upper()
        if frame == "NED":
            return (self.r_ecef_to_ned @ v).astype(float)
        if frame == "ENU":
            return (self.r_ecef_to_enu @ v).astype(float)
        raise ValueError(f"Unsupported local_frame: {local_frame!r}")

    def point_to_ecef(self, p_local_m: Vector3D, *, local_frame: LocalTangentFrame) -> Vector3D:
        """Convert a local tangent point (meters, relative to the origin) to ECEF (meters)."""

        return self.origin_ecef + self.vector_to_ecef(p_local_m, local_frame=local_frame)

    def point_from_ecef(self, p_ecef_m: Vector3D, *, local_frame: LocalTangentFrame) -> Vector3D:
        """Convert an ECEF point (meters) to a local tangent point (meters, relative to the origin)."""

        delta = np.asarray(p_ecef_m, dtype=float).reshape(3) - self.origin_ecef
        return self.vector_from_ecef(delta, local_frame=local_frame)

    def to_ecef_matrix(self, r_local_from_frame: Array3x3, *, local_frame: LocalTangentFrame) -> Array3x3:
        """Convert a rotation matrix expressed in the local tangent plane to ECEF."""

        r = np.asarray(r_local_from_frame, dtype=float).reshape(3, 3)
        frame = local_frame.upper()
        if frame == "NED":
            return (self.r_ned_to_ecef @ r).astype(float)
        if frame == "ENU":
            return (self.r_enu_to_ecef @ r).astype(float)
        raise ValueError(f"Unsupported local_frame: {local_frame!r}")

    def to_ltp_matrix(self, r_ecef_from_frame: Array3x3, *, local_frame: LocalTangentFrame) -> Array3x3:
        """Convert a rotation matrix expressed in ECEF into a local tangent frame."""

        r = np.asarray(r_ecef_from_frame, dtype=float).reshape(3, 3)
        frame = local_frame.upper()
        if frame == "NED":
            return (self.r_ecef_to_ned @ r).astype(float)
        if frame == "ENU":
            return (self.r_ecef_to_enu @ r).astype(float)
        raise ValueError(f"Unsupported local_frame: {local_frame!r}")
