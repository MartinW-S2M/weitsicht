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

"""Helpers for converting coordinates to GeoJSON geometries."""

from shapely import geometry

from weitsicht.utils import ArrayNx3, Vector3D


def get_geojson(coordinates: Vector3D | ArrayNx3, geom_type: str) -> dict:
    """Create a GeoJSON geometry dictionary for coordinates of a given type.

    Supported geometry types are ``Point``, ``LineString`` and ``Polygon``.

    :param coordinates: Coordinates used to construct the geometry.
    :type coordinates: Vector3D | ArrayNx3
    :param geom_type: Geometry type string.
    :type geom_type: str
    :return: GeoJSON geometry mapping as returned by :func:`shapely.geometry.mapping`.
    :rtype: dict
    :raises ValueError: If ``geom_type`` is unsupported.
    :raises Exception: If Shapely cannot construct the requested geometry from ``coordinates``.
    """

    if geom_type == "Point":
        geom = geometry.Point(coordinates)
    elif geom_type == "LineString":
        geom = geometry.LineString(coordinates)
    elif geom_type == "Polygon":
        geom = geometry.Polygon(coordinates)
    else:
        raise ValueError(r"Geometry is not supported. Only 'Point', 'LineString', 'Polygon'")

    geojson = geometry.mapping(geom)

    return geojson
