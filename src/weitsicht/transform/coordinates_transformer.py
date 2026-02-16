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

"""Coordinate transformation helpers based on :mod:`pyproj`."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import overload

import numpy as np
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError, ProjError
from pyproj.transformer import TransformerGroup
from shapely import errors as shapely_errors
from shapely import geometry

from weitsicht import cfg
from weitsicht.exceptions import CoordinateTransformationError
from weitsicht.utils import ArrayNx2, ArrayNx3, Vector2D, Vector3D

__all__ = ["CoordinateTransformer"]

logger = logging.getLogger(__name__)

# The doc states that cached Versions can be used to speed up performance significantly
TransformerFromCRS = lru_cache(Transformer.from_crs)
TransformerFromPipeline = lru_cache(Transformer.from_pipeline)


def get_transformation_transformergroup(crs_source: CRS, crs_target: CRS) -> TransformerGroup:  # pragma: no cover
    """Create a :class:`pyproj.transformer.TransformerGroup` between two CRSs.

    Note: the pyproj documentation states that the returned objects are not thread-safe.

    :param crs_source: Source CRS.
    :type crs_source: CRS
    :param crs_target: Target CRS.
    :type crs_target: CRS
    :return: Transformer group for ``crs_source`` → ``crs_target``.
    :rtype: TransformerGroup
    :raises CoordinateTransformationError: If no suitable transformation can be established.
    """

    try:
        transformer = TransformerGroup(
            crs_source,
            crs_target,
            always_xy=True,
            allow_ballpark=cfg._ballpark_transformation,
        )
        if cfg._only_best_transformation and not transformer.best_available:
            raise CoordinateTransformationError(
                "Best Transformation is not available but only best transformations are allowed"
            )
        return transformer

        # self.transformer = Transformer.from_crs(self._crs, crs_target, always_xy=True,
        #                                        only_best=cfg._only_best_transformation,
        #                                        allow_ballpark=cfg._ballpark_transformation)
    except ProjError as e:
        raise CoordinateTransformationError("Transformation could not be established.") from e


class CoordinateTransformer:
    """Class to handle transformations of coordinates.
    Basically a little pyproj wrapper with basic functions needed in the package.
    """

    @overload
    def __init__(self, transformer: Transformer): ...
    @overload
    def __init__(self, transformer: list[Transformer]): ...
    def __init__(self, transformer: Transformer | list[Transformer]):
        """Create a transformer wrapper.

        :param transformer: A single pyproj transformer or a chain of transformers.
        :type transformer: Transformer | list[Transformer]
        """

        if isinstance(transformer, Transformer):
            self._transformers = [transformer]
        else:
            self._transformers: list[Transformer] = transformer

    @property
    def source_crs(self) -> CRS | None:
        """Return the source CRS of the transformer chain."""
        return self._transformers[0].source_crs

    @property
    def target_crs(self) -> CRS | None:
        """Return the target CRS of the transformer chain."""
        return self._transformers[-1].target_crs

    @classmethod
    def from_pipeline(cls, proj_pipeline: str):
        """Create a transformer from a PROJ pipeline definition.

        :param proj_pipeline: PROJ pipeline string.
        :type proj_pipeline: str
        :return: Transformer instance.
        :rtype: CoordinateTransformer
        :raises CoordinateTransformationError: If the pipeline cannot be established.
        """

        try:
            transformer = TransformerFromPipeline(proj_pipeline)

        except (CRSError, ProjError) as e:
            raise CoordinateTransformationError("Establishing PROJ PIPELINE not working") from e

        _obj = cls(transformer=[transformer])
        return _obj

    @classmethod
    def from_crs(cls, crs_s: CRS | None, crs_t: CRS | None) -> CoordinateTransformer | None:
        """Create a coordinate transformer from source CRS to target CRS.

        If both CRSs are ``None`` or equal, this returns ``None`` (no transformation required).

        :param crs_s: Source CRS, or ``None``.
        :type crs_s: CRS | None
        :param crs_t: Target CRS, or ``None``.
        :type crs_t: CRS | None
        :return: Coordinate transformer or ``None`` if no transformation is required.
        :rtype: CoordinateTransformer | None
        :raises ValueError: If only one CRS is ``None``.
        :raises CoordinateTransformationError: If establishing a transformation fails.
        """

        # First we check if we anyhow need any transformation
        # If both CRS are None or both are equal we will not do any transformation
        if crs_s is None and crs_t is None:
            return None

        if (crs_s is None) != (crs_t is None):
            raise ValueError("Either both CRS have to be stated or for both None")

        assert crs_s is not None
        assert crs_t is not None
        if crs_s.equals(crs_t):
            return None

        transformer: list[Transformer] = []

        geod_s = crs_s.geodetic_crs.to_3d()
        geod_t = crs_t.geodetic_crs.to_3d()

        if geod_s.equals(geod_t):
            transformer.append(
                TransformerFromCRS(
                    crs_s,
                    crs_t,
                    always_xy=True,
                    allow_ballpark=cfg._ballpark_transformation,
                    only_best=cfg._only_best_transformation,
                )
            )

        else:
            try:
                transformer.append(
                    TransformerFromCRS(
                        crs_s,
                        geod_s,
                        always_xy=True,
                        allow_ballpark=cfg._ballpark_transformation,
                        only_best=cfg._only_best_transformation,
                    )
                )

                transformer.append(
                    TransformerFromCRS(
                        geod_s,
                        geod_t,
                        always_xy=True,
                        allow_ballpark=cfg._ballpark_transformation,
                        only_best=cfg._only_best_transformation,
                    )
                )

                transformer.append(
                    TransformerFromCRS(
                        geod_t,
                        crs_t,
                        always_xy=True,
                        allow_ballpark=cfg._ballpark_transformation,
                        only_best=cfg._only_best_transformation,
                    )
                )
            except (CRSError, ProjError) as e:
                raise CoordinateTransformationError("Establishing coordinate transformation for mapping failed") from e

        _obj = cls(transformer=transformer)

        return _obj

    @staticmethod
    def is_3d(coo: ArrayNx2 | ArrayNx3):
        """Return whether the coordinate array has a Z component.

        :param coo: Coordinate array.
        :type coo: ArrayNx2 | ArrayNx3
        :return: ``True`` if the input has shape N×3.
        :rtype: bool
        """
        if coo.shape[1] == 3:
            return True
        return False

    def transform(self, coordinates: ArrayNx3 | ArrayNx2, direction: str = "forward") -> ArrayNx3 | ArrayNx2:
        """Transform coordinates using the configured transformer chain.

        :param coordinates: Coordinates to transform as N×2 (2D) or N×3 (3D).
        :type coordinates: ArrayNx3 | ArrayNx2
        :param direction: Transform direction (``forward``, ``inverse``, ``identity``), defaults to ``forward``.
        :type direction: str
        :return: Transformed coordinates with the same dimensionality as input.
        :rtype: ArrayNx3 | ArrayNx2
        :raises CoordinateTransformationError: If input dimensions are invalid or a transformation fails.
        """

        _coordinates = coordinates.copy()
        if _coordinates.ndim == 1:
            _coordinates = np.array([_coordinates])

        if _coordinates.shape[1] < 2:
            raise CoordinateTransformationError("Wrong input dimension")

        # Iterate over the transformer list
        # if direction is inverse we need to start from end for list
        if direction == "inverse":
            transformers_order = list(reversed(self._transformers))
        else:
            transformers_order = self._transformers

        for trafo in transformers_order:
            try:
                # Other ways I called transform
                # res = trafo.transform(*_coordinates.T, errcheck=True)
                # p_crs = np.array(res).T
                # fx, fy, fz = transformer_srs_4979.transform(ray_start[:, 0], ray_start[:, 1],
                #                                            ray_start[:, 2], errcheck=True)

                if self.is_3d(_coordinates):
                    res = trafo.transform(
                        _coordinates[:, 0],
                        _coordinates[:, 1],
                        _coordinates[:, 2],
                        errcheck=True,
                        direction=direction,
                    )
                    _coordinates = np.array(res).T
                else:
                    res = trafo.transform(
                        _coordinates[:, 0],
                        _coordinates[:, 1],
                        errcheck=True,
                        direction=direction,
                    )
                    _coordinates = np.array(res).T

                if np.any(_coordinates == np.inf):
                    raise CoordinateTransformationError("Coordinate Transformation failed")

            except (CRSError, ProjError) as e:
                raise CoordinateTransformationError("Error while transforming coordinates") from e

        return _coordinates


class Geometries:  # pragma: no cover
    pass

    def __init__(self, points: ArrayNx3 | Vector3D | Vector2D | ArrayNx2):
        # points: ArrayNx3 | Vector3D | Vector2D | ArrayNx2):
        self._coordinates = points.copy()

        # The class uses internal only array with ndim 2. So point np.array([0,0,1]) will be np.array([[0,0,1]])
        if self._coordinates.ndim == 1:
            self._coordinates = np.array([self._coordinates])

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
