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

"""UTM conversion helpers."""

from __future__ import annotations

import logging

import numpy as np
from pyproj import CRS
from pyproj.crs.crs import CompoundCRS
from pyproj.exceptions import CRSError

from weitsicht.transform.coordinates_transformer import CoordinateTransformer

__all__ = ["get_zone", "is_wgs84_crs", "point_convert_utm_wgs84_egm2008","point_convert_utm_wgs84", "point_wgs84ell_to_utm"]

logger = logging.getLogger(__name__)


def get_zone(longitude: float, latitude: float) -> int:
    """Estimate the WGS84 utm zone from latitude and longitude.

    :param longitude: Longitude in degrees. Valid range: ``-180`` to ``180``.
    :type longitude: float
    :param latitude: Latitude in degrees. Valid range: ``-90`` to ``90``.
    :type latitude: float
    :return: UTM EPSG code.
    :rtype: int
    :raises ValueError: If limits are exceeded.
    """

    if abs(longitude) > 180 or abs(latitude) > 90:
        raise ValueError("Limits exceeded: -180<Longitude<180 and -90<Latitude<90")

    epsg_code_utm = int(32700 - np.round((45 + latitude) / 90, 0) * 100 + np.round((183 + longitude) / 6, 0))
    return epsg_code_utm


def is_wgs84_crs(crs_input: CRS | str | int) -> bool:
    """Return whether the CRS is a known WGS84 geodetic CRS.

    This intentionally only accepts EPSG:4326 and EPSG:4979, including equivalent WKT inputs.
    Projected CRS that use a WGS84 datum, such as UTM zones, return ``False``.
    """

    try:
        crs = CRS.from_user_input(crs_input)
    except (CRSError, ValueError):
        return False

    epsg_code = crs.to_epsg()
    if epsg_code in {4326, 4979}:
        return True

    return crs.equals(CRS.from_epsg(4326), ignore_axis_order=True) or crs.equals(
        CRS.from_epsg(4979), ignore_axis_order=True
    )


def point_convert_utm_wgs84_egm2008(
    crs_s: CRS, x: float, y: float, z: float
) -> tuple[float, float, float, CRS | CompoundCRS]:
    """Transform a single point into WGS84-UTM (EGM2008) coordinates.

    The point is first transformed to WGS84 3D (EPSG:4979), then assigned to a UTM zone and
    transformed to the corresponding compound CRS (UTM + EGM2008 geoid height).

    :param crs_s: CRS of the input point.
    :type crs_s: CRS
    :param x: X coordinate in ``crs_s`` units.
    :type x: float
    :param y: Y coordinate in ``crs_s`` units.
    :type y: float
    :param z: Z coordinate in ``crs_s`` units.
    :type z: float
    :return: Tuple ``(x_utm, y_utm, z_geoid, utm_crs)``.
    :rtype: tuple[float, float, float, CRS | CompoundCRS]
    :raises ValueError: If the transformed WGS84 point is outside UTM latitude/longitude limits.
    :raises CoordinateTransformationError: If a coordinate transformation cannot be established or applied.
    """
    return point_convert_utm_wgs84(crs_s, x, y, z, True)

def point_convert_utm_wgs84(
    crs_s: CRS, x: float, y: float, z: float, to_geoid_height: bool = True
) -> tuple[float, float, float, CRS | CompoundCRS]:
    """Transform a single point into WGS84-UTM coordinates.

    The input CRS has to be 3D so that the transformation is correct.
    The point is first transformed to WGS84 3D (EPSG:4979), then assigned to a UTM zone and
    transformed to the corresponding compound CRS (UTM + either ellipsoid or EGM2008 geoid height).

    :param crs_s: CRS of the input point.
    :type crs_s: CRS
    :param x: X coordinate in ``crs_s`` units.
    :type x: float
    :param y: Y coordinate in ``crs_s`` units.
    :type y: float
    :param z: Z coordinate in ``crs_s`` units.
    :type z: float
    :param to_geoid_height: If ``True``, the output will be in EGM2008 heights
    :type to_geoid_height: bool
    :return: Tuple ``(x_utm, y_utm, z_geoid, utm_crs)``.
    :rtype: tuple[float, float, float, CRS | CompoundCRS]
    :raises ValueError: If the transformed WGS84 point is outside UTM latitude/longitude limits.
    :raises CoordinateTransformationError: If a coordinate transformation cannot be established or applied.
    """

    crs_4979 = CRS(4979)

    coo_trafo = CoordinateTransformer.from_crs(crs_s, crs_4979)

    if coo_trafo is not None:
        coo_wgs84 = coo_trafo.transform(np.array([x, y, z]))
    else:
        coo_wgs84 = np.array([[x, y, z]])

    epsg_code_utm = get_zone(coo_wgs84[0][0], coo_wgs84[0][1])

    if to_geoid_height:
        utm_crs = CRS("EPSG:" + str(epsg_code_utm) + "+3855")
    else:
        # this will promote 2d UTM to 3D with ellipsoid heights
        utm_crs = CRS.from_epsg(epsg_code_utm).to_3d()

    # Here the trafo can actually never be None
    transformer_wgs84 = CoordinateTransformer.from_crs(crs_4979, utm_crs)

    assert transformer_wgs84 is not None
    coo_utm = transformer_wgs84.transform(*coo_wgs84)

    x_utm, y_utm, z_geoid = coo_utm[0, :]
    return x_utm, y_utm, z_geoid, utm_crs


def point_wgs84ell_to_utm(
    crs_s: CRS, lon_deg: float, lat_deg: float, h_m: float, to_geoid_height: bool = True
) -> tuple[float, float, float, CRS | CompoundCRS]:
    """Transform WGS84 ellipsoid coordinates to WGS84-UTM coordinates.
    The input CRS should be 3D so that the height can be transformed correct.

    The corresponding UTM zone of the point is assigned and then
    transformed to the corresponding compound CRS  (UTM + either ellipsoid or EGM2008 geoid height).
    This route directly converting to UTM, using other than WGS84 ellipsoid coordinates could lead to wrong results.

    point_wgs84ell_to_utm(CRS(4979), 16.0, 46.0,200,False) will transform that coordinates to
    WGS84-UTM coordinates and keep the ellipsoid heights.

    point_wgs84ell_to_utm(CRS(4979), 16.0, 46.0,200,True) will transform that coordinates to
    WGS84-UTM coordinates and transform the z (200m) which is given in ellipsoid heights into EGM2008 geoid heights.

    :param crs_s: CRS of the input point.
    :type crs_s: CRS
    :param lon_deg: Longitude coordinate in ``crs_s`` units.
    :type lon_deg: float
    :param lat_deg: Latitude coordinate in ``crs_s`` units.
    :type lat_deg: float
    :param h_m: Height in the CRS source system and units.
    :type h_m: float
    :param to_geoid_height: If ``True``, the output will be in EGM2008 heights
    :type to_geoid_height: bool
    :return: Tuple ``(x_utm, y_utm, z_geoid, utm_crs)``.
    :rtype: tuple[float, float, float, CRS | CompoundCRS]
    :raises ValueError: If the transformed WGS84 point is outside UTM latitude/longitude limits.
    :raises CoordinateTransformationError: If a coordinate transformation cannot be established or applied.
    """

    epsg_code_utm = get_zone(lon_deg, lat_deg)

    if to_geoid_height:
        utm_crs = CRS("EPSG:" + str(epsg_code_utm) + "+3855")
    else:
        utm_crs = CRS.from_epsg(epsg_code_utm).to_3d()

    # Here the trafo can actually never be None
    transformer_wgs84 = CoordinateTransformer.from_crs(crs_s=crs_s, crs_t=utm_crs)

    coo_wgs84 = np.array([[lon_deg, lat_deg, h_m]])
    assert transformer_wgs84 is not None
    coo_utm = transformer_wgs84.transform(coo_wgs84)

    x_utm, y_utm, z = coo_utm[0, :]
    return x_utm, y_utm, z, utm_crs
