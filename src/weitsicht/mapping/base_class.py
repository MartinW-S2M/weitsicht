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

"""Base interfaces for mapping backends.

Mapping backends implement the conversion between rays/coordinates and mapped 3D coordinates
in a target domain (e.g. a horizontal plane, a raster DEM, a mesh).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from enum import Enum
from typing import Any, TypeVar

from pyproj import CRS

from weitsicht.transform.coordinates_transformer import CoordinateTransformer
from weitsicht.utils import ArrayNx2, ArrayNx3, MappingResult

__all__ = ["MappingType", "MappingBase"]


class MappingType(Enum):
    """Enum identifying available mapping backends.

    The :attr:`fullname` attribute is a stable, human-readable identifier used for
    (de-)serialization (e.g. in :meth:`~weitsicht.mapping.base_class.MappingBase.param_dict`).
    """

    fullname: str  # human-readable identifier set on each enum member
    HorizontalPlane = 1, "horizontalPlane"
    Raster = 2, "Raster"
    GeoreferencedNumpyArray = 2, "GeorefArray"
    Trimesh = 3, "Trimesh"

    def __new__(cls, value: int, name: str):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


_Self = TypeVar("_Self", bound="MappingBase")


class MappingBase:
    """Abstract base class for all mapping backends."""

    def __init__(self):
        """Initialize a mapper with an optional CRS (set via :attr:`crs`)."""
        self._crs: CRS | None = None

    @classmethod
    @abstractmethod
    def from_dict(cls: type[_Self], mapper_dict: Mapping[str, Any]) -> _Self:
        """Create a mapper instance from a configuration dictionary.

        :param mapper_dict: Mapper configuration dictionary (typically created via ``mapper.param_dict``).
        :type mapper_dict: Mapping[str, Any]
        :return: Instantiated mapper.
        :rtype: MappingBase
        :raises KeyError: If a required dictionary key is missing.
        :raises ValueError: If configuration values are invalid.
        :raises CRSInputError: If CRS input in the configuration is invalid (e.g. malformed WKT).
        :raises CRSnoZaxisError: If a required CRS does not define a Z axis.
        :raises MappingError: If the mapper cannot be initialized.
        """
        pass

    @property
    @abstractmethod
    def type(self) -> MappingType:
        """Return the mapper type.

        :return: Mapper type.
        :rtype: MappingType
        """
        pass

    @property
    @abstractmethod
    def param_dict(self) -> Mapping[str, Any]:
        """Return the mapper configuration dictionary.

        This dictionary is intended to be used with :meth:`from_dict`.

        :return: Mapper configuration dictionary.
        :rtype: Mapping[str, Any]
        """
        pass

    @property
    def crs(self) -> CRS | None:
        """Return the CRS of the mapper (if set).

        :return: CRS of the mapper.
        :rtype: CRS | None
        """
        return self._crs

    @crs.setter
    def crs(self, crs: CRS | None):
        """Set the CRS of the mapper.

        :param crs: CRS to use, or ``None`` to unset.
        :type crs: CRS | None
        """
        self._crs = crs

    @property
    def crs_wkt(self) -> str | None:
        """Return the CRS as WKT string, or ``None`` if not set.

        :return: CRS WKT string.
        :rtype: str | None
        """
        if self._crs is not None:
            return self._crs.to_wkt()
        return None

    @abstractmethod
    def map_coordinates_from_rays(
        self,
        ray_vectors_crs_s: ArrayNx3,
        ray_start_crs_s: ArrayNx3,
        crs_s: CRS | None = None,
        transformer: CoordinateTransformer | None = None,
    ) -> MappingResult:
        """Map 3D coordinates from ray definitions.

        The input rays are provided by a ray start point and a ray direction vector. If a CRS
        conversion is required, pass either ``crs_s`` (and the mapper's :attr:`crs` must be set)
        or a preconfigured ``transformer``.

        :param ray_vectors_crs_s: Ray direction vectors (N×3).
        :type ray_vectors_crs_s: ArrayNx3
        :param ray_start_crs_s: Ray start points (N×3), same shape as ``ray_vectors_crs_s``.
        :type ray_start_crs_s: ArrayNx3
        :param crs_s: CRS of the input rays, defaults to None.
        :type crs_s: CRS | None
        :param transformer: Coordinate transformer to mapper CRS, defaults to None.
        :type transformer: CoordinateTransformer | None
        :return: Mapping result containing mapped coordinates and a validity mask.
        :rtype: MappingResult
        :raises ValueError: If input arrays have incompatible shapes.
        :raises CRSInputError: If both ``crs_s`` and ``transformer`` are provided.
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """
        pass

    @abstractmethod
    def map_heights_from_coordinates(
        self,
        coordinates_crs_s: ArrayNx3 | ArrayNx2,
        crs_s: CRS | None = None,
        transformer: CoordinateTransformer | None = None,
    ) -> MappingResult:
        """Sample heights for given 2D/3D coordinates.

        This is typically used to lift planar coordinates (x, y) to 3D (x, y, z) using a DEM.

        :param coordinates_crs_s: Coordinates to sample (N×2 or N×3). If N×2, Z is backend-defined.
        :type coordinates_crs_s: ArrayNx3 | ArrayNx2
        :param crs_s: CRS of the input coordinates, defaults to None.
        :type crs_s: CRS | None
        :param transformer: Coordinate transformer to mapper CRS, defaults to None.
        :type transformer: CoordinateTransformer | None
        :return: Mapping result containing coordinates with sampled heights and a validity mask.
        :rtype: MappingResult
        :raises ValueError: If input arrays have incompatible shapes.
        :raises CRSInputError: If both ``crs_s`` and ``transformer`` are provided.
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """

        pass
