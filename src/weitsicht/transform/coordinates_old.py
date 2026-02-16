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

"""Deprecated coordinate transformation utilities.

This module contains a legacy transformer implementation kept for information.
For new code, prefer :class:`weitsicht.transform.coordinates_transformer.CoordinateTransformer`.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError, ProjError
from shapely import errors as shapely_errors
from shapely import geometry

from weitsicht import cfg
from weitsicht.exceptions import CoordinateTransformationError
from weitsicht.utils import ArrayNx2, ArrayNx3, Vector2D, Vector3D

logger = logging.getLogger(__name__)


# The doc states that cached Versions can be used to speed up performance significantly
TransformerFromCRS = lru_cache(Transformer.from_crs)
TransformerFromPipeline = lru_cache(Transformer.from_pipeline)


class CoordinatesTransformer:  # pragma: no cover
    """Legacy coordinate transformation wrapper around :mod:`pyproj`.

    The coordinates of this class are read-only once initialized.
    """

    def __init__(self, crs: CRS, points: ArrayNx3 | Vector3D | Vector2D | ArrayNx2):
        """Create a transformer with CRS and coordinates.

        :param crs: CRS of the input coordinates.
        :type crs: CRS
        :param points: Coordinates to transform.
        :type points: ArrayNx3 | Vector3D | Vector2D | ArrayNx2
        """

        self._coordinates = points.copy()

        # The class uses internal only array with ndim 2. So point np.array([0,0,1]) will be np.array([[0,0,1]])
        if self._coordinates.ndim == 1:
            self._coordinates = np.array([self._coordinates])

        # self._coordinates.setflags(write=False)
        self._crs: CRS = crs

        self.transformer = None

    @classmethod
    def from_crs(
        cls, crs_s: CRS, crs_t: CRS, points: ArrayNx3 | Vector3D | Vector2D | ArrayNx3
    ) -> CoordinatesTransformer:
        """Create and immediately transform coordinates from ``crs_s`` to ``crs_t``.

        :param crs_s: Source CRS.
        :type crs_s: CRS
        :param crs_t: Target CRS.
        :type crs_t: CRS
        :param points: Coordinates to transform.
        :type points: ArrayNx3 | Vector3D | Vector2D | ArrayNx3
        :return: Transformer instance in target CRS.
        :rtype: CoordinatesTransformer
        :raises CoordinateTransformationError: If establishing or applying the transformation fails.
        """

        _obj = cls(crs=crs_s, points=points)

        return _obj.to_crs(crs_t)

    @property
    def is_point(self) -> bool:
        if self._coordinates.shape[0] == 1:
            return True
        return False

    @property
    def is_3d(self):
        if self._coordinates.shape[1] == 3:
            return True
        return False

    def geojson(self, geom_type: str) -> dict:
        """Convert the stored coordinates into a GeoJSON mapping.

        :param geom_type: Geometry type (``Point``, ``LineString``, ``Polygon``).
        :type geom_type: str
        :return: GeoJSON mapping dict.
        :rtype: dict
        :raises CoordinateTransformationError: If geometry type is unsupported or coordinates are invalid.
        """

        try:
            if geom_type == "Point":
                geojson = geometry.Point(self.coordinates)
            elif geom_type == "LineString":
                geojson = geometry.LineString(self._coordinates)
            elif geom_type == "Polygon":
                geojson = geometry.Polygon(self._coordinates)
            else:
                raise CoordinateTransformationError(r"Geometry is not supported. Only 'Point', 'LineString', 'Polygon'")

        except (shapely_errors.GEOSException, ValueError) as err:
            # Will also raise if multi coordinates and point type is used
            raise CoordinateTransformationError("GeoJSON can not be established from given coordinates") from err

        geojson = geometry.mapping(geojson)

        return geojson

    @property
    def coordinates(self) -> ArrayNx3 | ArrayNx2:
        return self._coordinates

    @coordinates.setter
    def coordinates(self, new_coordinate: ArrayNx3 | ArrayNx2):

        points = new_coordinate.copy()

        # The class uses internal only array with ndim 2. So point np.array([0,0,1]) will be np.array([[0,0,1]])
        if points.ndim == 1:
            new_coordinate = np.array([new_coordinate])
        self._coordinates = new_coordinate

    def to_crs(self, crs_target: CRS) -> CoordinatesTransformer:
        """Transform the stored coordinates to ``crs_target``.

        :param crs_target: Target CRS.
        :type crs_target: CRS
        :return: New transformer instance holding coordinates in ``crs_target``.
        :rtype: CoordinatesTransformer
        :raises CoordinateTransformationError: If establishing or applying the transformation fails.
        """

        # If the same coordinate systems are used we will spare the trafo
        if self._crs.equals(crs_target):
            return CoordinatesTransformer(crs_target, self._coordinates)

        try:
            self.transformer = TransformerFromCRS(
                self._crs,
                crs_target,
                always_xy=True,
                allow_ballpark=cfg._ballpark_transformation,
                only_best=cfg._only_best_transformation,
            )
            # if cfg._only_best_transformation and not tr_group.best_available:
            #    raise CoordinateTransformationError("Best Transformation is not available "
            #                                        "but only best transformations are allowed")
            # self.transformer = tr_group.transformers[0]

            # self.transformer = Transformer.from_crs(self._crs, crs_target, always_xy=True,
            #                                        only_best=cfg._only_best_transformation,
            #                                        allow_ballpark=cfg._ballpark_transformation)
        except ProjError as e:
            raise CoordinateTransformationError("Transformation could not be established.") from e

        try:
            if self.is_3d:
                fx, fy, fz = self.transformer.transform(
                    self._coordinates[:, 0],
                    self._coordinates[:, 1],
                    self._coordinates[:, 2],
                    errcheck=True,
                )
                p_crs = np.vstack((fx, fy, fz)).T
            else:
                fx, fy = self.transformer.transform(self._coordinates[:, 0], self._coordinates[:, 1], errcheck=True)
                p_crs = np.vstack((fx, fy)).T

            new_trafo = CoordinatesTransformer(crs_target, p_crs)
            new_trafo.transformer = self.transformer
            return new_trafo

        except (CRSError, ProjError) as e:
            raise CoordinateTransformationError("Error while transforming coordinates") from e
