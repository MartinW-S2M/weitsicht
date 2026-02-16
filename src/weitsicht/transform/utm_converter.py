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

from weitsicht.transform.coordinates_transformer import CoordinateTransformer

__all__ = ["get_zone", "point_convert_utm_wgs84_egm2008"]

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

    crs_4979 = CRS(4979)

    coo_trafo = CoordinateTransformer.from_crs(crs_s, crs_4979)

    if coo_trafo is not None:
        coo_wgs84 = coo_trafo.transform(np.array([x, y, z]))
    else:
        coo_wgs84 = np.array([[x, y, z]])

    epsg_code_utm = get_zone(coo_wgs84[0][0], coo_wgs84[0][1])

    utm_crs = CRS("EPSG:" + str(epsg_code_utm) + "+3855")

    # Here the trafo can actually never be None
    transformer_wgs84 = CoordinateTransformer.from_crs(crs_4979, utm_crs)

    assert transformer_wgs84 is not None
    coo_utm = transformer_wgs84.transform(*coo_wgs84)

    x_utm, y_utm, z_geoid = coo_utm[0, :]
    return x_utm, y_utm, z_geoid, utm_crs
