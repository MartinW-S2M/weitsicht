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

"""Base types and interfaces for image models."""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum

from pyproj import CRS

from weitsicht.geometry.coo_geojson import get_geojson
from weitsicht.mapping.base_class import MappingBase
from weitsicht.transform.coordinates_transformer import CoordinateTransformer
from weitsicht.utils import (
    ArrayNx2,
    ArrayNx3,
    MappingResult,
    ProjectionResult,
    Vector3D,
)

__all__ = ["ImageType", "ImageBase"]


class ImageType(Enum):
    """Enum of supported image model types.

    The ``fullname`` attribute is used for serialization (e.g. in ``param_dict``)
    and must match what factory functions expect (e.g. ``get_image_from_dict``).
    Add new image model types here.
    """

    fullname: str
    Unknown = 0, "unknown"
    Perspective = 1, "perspective"
    Orthophoto = 2, "ortho"

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class ImageBase:
    """Base class for image models.

    This base class defines a common API for projecting 3D coordinates into image
    pixel coordinates and mapping image pixels to 3D coordinates (via an optional
    :class:`~weitsicht.mapping.base_class.MappingBase`).

    Coordinate system conventions:
    - Image pixel CRS: x points right, y points down (origin at top-left pixel edge).
    - World CRS: any cartesian CRS; for mapping it is usually best to use a projected CRS.

    Subclasses must implement:
    - `from_dict`
    - `type`
    - `param_dict`
    - `is_geo_referenced`
    - `center`
    - `position_to_crs`
    - `project`
    - `map_center_point`
    - `map_footprint`
    - `map_points`
    """

    def __init__(
        self,
        width: float | int,
        height: float | int,
        mapper: MappingBase | None = None,
        crs: CRS | None = None,
    ):
        """Initialize the image base class.

        :param width: Image width in pixels.
        :type width: float | int
        :param height: Image height in pixels.
        :type height: float | int
        :param mapper: Mapping instance, defaults to ``None``.
        :type mapper: MappingBase | None
        :param crs: World CRS of the image, defaults to ``None``.
        :type crs: CRS | None
        """

        self._crs: CRS | None = crs
        self.mapper = mapper

        self._width: int = int(width)
        self._height: int = int(height)

        self._position = None

    @property
    def mapper(self) -> MappingBase | None:
        # get mapper
        return self._mapper

    @mapper.setter
    def mapper(self, mapper: MappingBase | None):
        self._mapper = mapper

    @classmethod
    @abstractmethod
    def from_dict(cls, param_dict: dict, mapper: MappingBase | None = None) -> ImageBase:
        """Create an image model from a parameter dictionary.

        Implementations should accept the dictionary returned by :attr:`param_dict`.

        :param param_dict: Dictionary with image parameters.
        :type param_dict: dict
        :param mapper: Mapping instance, defaults to ``None``.
        :type mapper: MappingBase | None
        :return: Image model instance.
        :rtype: ImageBase
        :raises KeyError: If required keys are missing.
        :raises ValueError: If configuration values are invalid.
        :raises TypeError: If configuration values have incompatible types.
        """
        pass

    @property
    @abstractmethod
    def type(self) -> ImageType:
        """Return the image model type.

        :return: Image type.
        :rtype: ImageType
        """
        pass

    @property
    @abstractmethod
    def param_dict(self) -> dict:
        """Return image parameters as a dictionary.

        The returned dictionary must be compatible with :meth:`from_dict`.

        :return: Image parameters.
        :rtype: dict
        """
        pass

    @property
    @abstractmethod
    def is_geo_referenced(self) -> bool:
        """Return whether the image is geo-referenced.

        :return: ``True`` if geo-referenced (and mapping/projection can be computed), otherwise ``False``.
        :rtype: bool
        """
        pass

    @property
    @abstractmethod
    def center(self) -> tuple[float, float]:
        """Return the center of the image in pixel coordinates.

        Implementations may define the center differently (e.g. the principal point
        for perspective cameras).

        :return: Center point (x, y) in pixel coordinates.
        :rtype: tuple[float, float]
        """
        pass

    @abstractmethod
    def position_to_crs(self, crs_t: CRS) -> Vector3D | None:
        """Return the image reference position in the target CRS.

        :param crs_t: Target CRS.
        :type crs_t: CRS
        :return: Position in ``crs_t`` or ``None`` if unavailable.
        :rtype: Vector3D | None
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """
        pass

    @abstractmethod
    def project(
        self, coordinates: ArrayNx3, crs_s: CRS | None = None, transformer: CoordinateTransformer | None = None
    ) -> ProjectionResult:
        """Project 3D coordinates into image pixel coordinates.

        :param coordinates: 3D coordinates to project.
        :type coordinates: ArrayNx3
        :param crs_s: CRS of the input coordinates, defaults to ``None``.
        :type crs_s: CRS | None
        :param transformer: Optional coordinate transformer, defaults to ``None``.
        :type transformer: CoordinateTransformer | None
        :return: Result of the projection.
        :rtype: ProjectionResult
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """
        pass

    @abstractmethod
    def map_center_point(
        self, mapper: MappingBase | None = None, transformer: CoordinateTransformer | None = None
    ) -> MappingResult:
        """Map the image center pixel to 3D coordinates.

        For perspective images this is typically the principal point.

        :param mapper: Mapper to use, defaults to ``None`` (uses :attr:`mapper`).
        :type mapper: MappingBase | None
        :param transformer: Optional coordinate transformer, defaults to ``None``.
        :type transformer: CoordinateTransformer | None
        :return: Mapping result.
        :rtype: MappingResult
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """
        pass

    @abstractmethod
    def map_footprint(
        self,
        points_per_edge: int = 0,
        mapper: MappingBase | None = None,
        transformer: CoordinateTransformer | None = None,
    ) -> MappingResult:
        """Map footprint of image to 3D.

        :param points_per_edge: The number of points inserted between corners, defaults to ``0``.
            ``0`` means only corner points are mapped.
        :type points_per_edge: int
        :param mapper: Mapper to use, defaults to ``None`` (uses :attr:`mapper`).
        :type mapper: MappingBase | None
        :param transformer: Optional coordinate transformer, defaults to ``None``.
        :type transformer: CoordinateTransformer | None
        :return: Mapping result.
        :rtype: MappingResult
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """

        pass

    @abstractmethod
    def map_points(
        self,
        points_image: ArrayNx2 | ArrayNx3 | list[list[float]] | list[list[int]] | list[float] | list[int],
        mapper: MappingBase | None = None,
        transformer: CoordinateTransformer | None = None,
    ) -> MappingResult:
        """Map image pixel coordinates to 3D coordinates.

        :param points_image: Pixel coordinates.
        :type points_image: ArrayNx2 | ArrayNx3 | list[list[float]] | list[list[int]] | list[float] | list[int]
        :param mapper: Mapper to use, defaults to ``None`` (uses :attr:`mapper`).
        :type mapper: MappingBase | None
        :param transformer: Optional coordinate transformer, defaults to ``None``.
        :type transformer: CoordinateTransformer | None
        :return: Mapping result.
        :rtype: MappingResult
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """

        pass

    # Standard properties
    @property
    def position_wgs84(self) -> tuple[float, float, float] | None:
        """Return the position in WGS84 (EPSG:4979) of the image or None if not possible.
        The position_to_crs method is defined by the class implementation.

        :return: Position in WGS84 (x, y, z) or ``None``.
        :rtype: tuple[float, float, float] | None
        """

        coordinates = self.position_to_crs(CRS.from_epsg(4979))

        if coordinates is not None:
            pos_wgs84 = coordinates
            return float(pos_wgs84[0]), float(pos_wgs84[1]), float(pos_wgs84[2])

        return None

    @property
    def position_wgs84_geojson(self) -> dict | None:
        """Return the image position in WGS84 (EPSG:4979) as a GeoJSON point.

        :return: GeoJSON ``Point`` mapping or ``None`` if the position is unavailable.
        :rtype: dict | None
        """
        coordinates = self.position_to_crs(CRS.from_epsg(4979))
        if coordinates is not None:
            return get_geojson(coordinates, "Point")
        return None

    @property
    def crs(self) -> CRS | None:
        """Return the image world CRS.

        :return: CRS of the image or ``None`` if unknown.
        :rtype: CRS | None
        """
        return self._crs

    # TODO should we actually perform a transformation?
    # Then we would need orientation matrix to be transformed accordingly
    @crs.setter
    def crs(self, crs: CRS):
        """Set images CRS from pyproj CRS object

        :param crs: CRS of the image.
        :type crs: CRS
        """
        self._crs = crs

    @property
    def crs_wkt(self) -> str | None:
        """Return images CRS as wkt string"""
        if self._crs is not None:
            return self._crs.to_wkt()
        return None

    @property
    def crs_proj4(self) -> str | None:
        """Return images CRS as proj4 string"""
        if self._crs is not None:
            return self._crs.to_proj4()
        return None

    @property
    def width(self) -> int:
        """Return width of image (column number)"""
        return self._width

    @property
    def height(self) -> int:
        """Return height of image (row number)"""
        return self._height

    @property
    def shape(self) -> tuple[int, int]:
        """Return shape of image as tuple(width, height)"""
        return self._width, self._height
