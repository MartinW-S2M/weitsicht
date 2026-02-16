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


"""Orthophoto image model backed by rasterio geotransforms."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pyproj.exceptions
import rasterio
from affine import Affine
from pyproj import CRS
from rasterio import errors
from shapely import geometry

from weitsicht.exceptions import (
    CRSInputError,
    MapperMissingError,
    NotGeoreferencedError,
)
from weitsicht.image.base_class import ImageBase, ImageType
from weitsicht.mapping.base_class import MappingBase
from weitsicht.transform.coordinates_transformer import CoordinateTransformer
from weitsicht.utils import (
    ArrayNx2,
    ArrayNx3,
    Issue,
    MappingResult,
    MappingResultSuccess,
    MaskN_,
    ProjectionResult,
    ProjectionResultSuccess,
    ResultFailure,
    Vector3D,
    to_array_nx2,
)

__all__ = ["ImageOrtho"]

logger = logging.getLogger(__name__)


class ImageOrtho(ImageBase):
    """Orthophoto image model based on an affine geotransform.

    This image type uses an affine transform to convert between image pixel coordinates and world
    coordinates on a plane (x/y). Heights can optionally be provided via a mapper.
    """

    def __init__(
        self,
        width: float | int,
        height: float | int,
        geo_transform: rasterio.Affine | None = None,
        resolution: float = 0.0,
        crs: CRS | None = None,
        mapper: MappingBase | None = None,
    ):
        """Initialize an orthophoto image model.

        :param width: Image width in pixels.
        :type width: float | int
        :param height: Image height in pixels.
        :type height: float | int
        :param geo_transform: Affine geotransform, defaults to ``None``.
        :type geo_transform: rasterio.Affine | None
        :param resolution: Pixel resolution in world units, defaults to ``0.0``. If ``0.0`` and a geotransform is
            provided, the mean scaling of the transform is used.
        :type resolution: float
        :param crs: World CRS of the orthophoto, defaults to ``None``.
        :type crs: CRS | None
        :param mapper: Mapping instance used for height queries, defaults to ``None``.
        :type mapper: MappingBase | None
        """
        super().__init__(mapper=mapper, crs=crs, width=width, height=height)

        # geo transform
        self.geo_transform = geo_transform

        self._res = resolution

        if self._res == 0.0:
            if self.geo_transform is not None:
                # TODO all that resolution could be optimized, problem can arise if non-cartesian CRS is given
                # for the ortho imagery
                self._res = float(np.mean(self.geo_transform._scaling))

    @classmethod
    def from_file(cls, path: Path | str, crs: CRS | None = None, mapper: MappingBase | None = None) -> ImageOrtho:
        """Create an orthophoto image model from a raster file.

        :param path: Path to the raster file.
        :type path: Path | str
        :param crs: CRS which overrides the file CRS, defaults to ``None``.
        :type crs: CRS | None
        :param mapper: Mapping instance, defaults to ``None``.
        :type mapper: MappingBase | None
        :return: Image model instance.
        :rtype: ImageOrtho
        :raises Exception: If rasterio cannot read the file or required metadata is unavailable.
        """

        if not Path(path).exists():
            raise FileNotFoundError("Specified file does not exist")
        path = Path(path)

        dataset = rasterio.open(path)
        width = dataset.shape[0]
        height = dataset.shape[1]

        # rasterio returns identity if file has no geo-reference
        # We want that gt is None instead of having an identity matrix
        if dataset.transform.is_identity:
            gt = None
        else:
            gt = dataset.transform

        _crs: CRS | None = crs
        if crs is None:
            # get crs code from orthophoto and promote to 3d.
            # Actually that would not be needed for the orthophoto
            # but to be sure all pyproj transformations are working
            if dataset.crs is not None:
                _crs = CRS(dataset.crs).to_3d()

        # TODO this should be optimized
        if dataset.crs is not None:
            try:
                linear_unit_factor = dataset.crs.linear_units_factor
                res = float(np.mean(dataset.res) * linear_unit_factor[1])
            except errors.CRSError:
                linear_unit_factor = 1.0
                res = float(np.mean(dataset.res) * linear_unit_factor)

        else:
            res = float(np.mean(dataset.res))

        # initialize the image class
        image = cls(width, height, geo_transform=gt, resolution=res, crs=_crs, mapper=mapper)

        return image

    @property
    def resolution(self) -> float:
        """Return the pixel resolution in world units.

        :return: Resolution value.
        :rtype: float
        """
        return self._res

    @property
    def type(self):
        """Return the image model type.

        :return: Image type.
        :rtype: ImageType
        """
        return ImageType.Orthophoto

    @classmethod
    def from_dict(cls, param_dict: dict, mapper: MappingBase | None = None) -> ImageOrtho:
        """Create an :class:`ImageOrtho` from a parameter dictionary.

        Required keys are:
        - ``width``
        - ``height``

        Optional keys are:
        - ``geo_transform`` (Affine as dict)
        - ``resolution``
        - ``crs`` (WKT)

        :param param_dict: Dictionary with image parameters.
        :type param_dict: dict
        :param mapper: Mapping instance, defaults to ``None``.
        :type mapper: MappingBase | None
        :return: Image model instance.
        :rtype: ImageOrtho
        :raises KeyError: If required keys are missing.
        :raises ValueError: If values are invalid.
        :raises TypeError: If values have incompatible types.
        :raises CRSInputError: If the CRS WKT string is invalid or unsupported.
        """

        width = param_dict["width"]
        height = param_dict["height"]

        # get geo transform
        gt = None
        if param_dict.get("geo_transform", None) is not None:
            gt = rasterio.Affine(**param_dict["geo_transform"])

        resolution = param_dict.get("resolution", 0.0)

        crs = None
        if param_dict.get("crs", False):
            try:
                crs = CRS(param_dict["crs"])
            except pyproj.exceptions.CRSError as err:
                raise CRSInputError("No valid CRS WKT string supported") from err

        image = cls(
            width,
            height,
            geo_transform=gt,
            resolution=resolution,
            crs=crs,
            mapper=mapper,
        )

        return image

    @property
    def param_dict(self) -> dict:
        """Return image parameters as a dictionary.

        The returned dictionary is compatible with :meth:`from_dict`.

        :return: Image parameters.
        :rtype: dict
        """
        param_dict = {
            "type": self.type.fullname,
            "width": self._width,
            "height": self._height,
            "geo_transform": self.geo_transform._asdict() if self.geo_transform is not None else None,
            "resolution": self._res,
        }
        if self.crs is not None:
            param_dict["crs"] = self.crs.to_wkt()

        return param_dict

    @property
    def center(self) -> tuple[float, float]:
        """Return the image center in pixel coordinates.

        :return: Center point (x, y) in pixel coordinates.
        :rtype: tuple[float, float]
        """
        return self.width / 2.0, self.height / 2.0

    @property
    def is_geo_referenced(self) -> bool:
        """Return whether the image is geo-referenced.

        :return: ``True`` if a geotransform is set, otherwise ``False``.
        :rtype: bool
        """
        if self.geo_transform is not None:
            return True
        return False

    def image_points_inside(self, point_image_coordinates: ArrayNx2) -> MaskN_:
        """Return whether image pixel coordinates are inside the raster bounds.

        :param point_image_coordinates: Image pixel coordinates (image CRS).
        :type point_image_coordinates: ArrayNx2
        :return: Boolean mask of shape ``(N,)``.
        :rtype: ``MaskN_``
        :raises ValueError: If the input cannot be converted to an ``(N, 2)`` array.
        :raises TypeError: If the input has incompatible types.
        """

        # Format points to be numpy array Nx2
        point_image_coordinates = to_array_nx2(point_image_coordinates)

        min_border = np.logical_and(point_image_coordinates[:, 0] >= 0, point_image_coordinates[:, 1] >= 0)
        max_border = np.logical_and(
            point_image_coordinates[:, 0] < (self.width - 1.0),
            point_image_coordinates[:, 1] < (self.height - 1.0),
        )
        valid_index = np.logical_and(min_border, max_border)

        return valid_index

    def position_to_crs(self, crs_t: CRS, transformer: CoordinateTransformer | None = None) -> Vector3D | None:
        """Return the orthophoto center position in the target CRS.

        If a mapper is available, the Z value is obtained by mapping the center point.

        :param crs_t: Target CRS.
        :type crs_t: CRS
        :param transformer: Optional coordinate transformer, defaults to ``None``.
        :type transformer: CoordinateTransformer | None
        :return: Center position in ``crs_t`` or ``None`` if unavailable.
        :rtype: Vector3D | None
        :raises CRSInputError: If CRS/transformer input is invalid.
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """

        if self.geo_transform is None:
            return None

        point_center_2d = self.geo_transform * [self.width / 2.0, self.height / 2.0]

        point_center_3d = np.array([[point_center_2d[0], point_center_2d[1], 0]])

        if self.mapper is not None:
            crs_arg = None if transformer is not None else self._crs
            mapping_result = self.mapper.map_heights_from_coordinates(point_center_2d, crs_arg, transformer=transformer)

            if mapping_result.ok is True:
                point_center_3d = mapping_result.coordinates

        # Transformer
        if self._crs is not None and transformer is not None:
            raise CRSInputError("Either crs or transformation can be used or both None")
        coo_trafo = (
            transformer if transformer is not None else CoordinateTransformer.from_crs(crs_s=self._crs, crs_t=crs_t)
        )

        if coo_trafo is not None:
            coordinates_crs_t = coo_trafo.transform(point_center_3d)
        else:
            coordinates_crs_t = np.array([[point_center_3d[0][0], point_center_3d[0][1], point_center_3d[0][2]]])

        return coordinates_crs_t[0, :]

    def project(
        self, coordinates: ArrayNx3 | ArrayNx2, crs_s: CRS | None = None, transformer: CoordinateTransformer | None = None
    ) -> ProjectionResult:
        """Project 3D coordinates into orthophoto pixel coordinates.

        Once the coordinates from the coordinates crs (crs_s) are transformed to the mapper crs,
        then onlyx/y coordinates are used for the affine transform; z is ignored.

        The reason of the tranformatoin from crs_s to mapper crs is why also here for orthophoto,
        3D coordiantes must be used as input.

        :param coordinates: 3D coordinates to project.
        :type coordinates: ArrayNx3
        :param crs_s: CRS of the input coordinates, defaults to ``None``.
        :type crs_s: CRS | None
        :param transformer: Optional coordinate transformer, defaults to ``None``.
        :type transformer: CoordinateTransformer | None
        :return: Projection result.
        :rtype: ProjectionResult
        :raises ValueError: If CRS/transformer inputs are inconsistent.
        :raises NotGeoreferencedError: If the image is not geo-referenced.
        :raises CRSInputError: If CRS/transformer input is invalid.
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """

        if not self.is_geo_referenced:
            raise NotGeoreferencedError()

        # _coordinates = to_array_nx3(coordinates)

        # Transformer
        if crs_s is not None and transformer is not None:
            raise CRSInputError("Either crs or transformation can be used or both None")
        coo_trafo = (
            transformer if transformer is not None else CoordinateTransformer.from_crs(crs_s=crs_s, crs_t=self._crs)
        )

        if coo_trafo is not None:
            point_3d_crs = coo_trafo.transform(coordinates)
        else:
            point_3d_crs = coordinates * 1.0

        assert self.geo_transform is not None

        gt = cast(Affine, ~self.geo_transform)  # invert
        x, y = cast(tuple[np.ndarray, np.ndarray], gt * (point_3d_crs[:, 0], point_3d_crs[:, 1]))
        # x, y = ~self.geo_transform * (point_3d_crs[:, 0], point_3d_crs[:, 1])

        raster_coordinates = np.vstack((x, y)).T
        valid_mask = self.image_points_inside(raster_coordinates)

        issue = set()
        if not np.all(valid_mask):
            issue = {Issue.INVALID_PROJECTIIONS}
            if not np.any(valid_mask):
                return ResultFailure(
                    ok=False,
                    error="None of the coordinates are on the raster",
                    issues={Issue.INVALID_PROJECTIIONS},
                )

        return ProjectionResultSuccess(ok=True, pixels=raster_coordinates, mask=valid_mask, issues=issue)

    def map_center_point(
        self, mapper: MappingBase | None = None, transformer: CoordinateTransformer | None = None
    ) -> MappingResult:
        """Map the orthophoto center pixel to 3D coordinates via a mapper.

        :param mapper: Mapper to use, defaults to ``None`` (uses :attr:`~weitsicht.image.base_class.ImageBase.mapper`).
        :type mapper: MappingBase | None
        :param transformer: Optional coordinate transformer, defaults to ``None``.
        :type transformer: CoordinateTransformer | None
        :return: Mapping result.
        :rtype: MappingResult
        :raises NotGeoreferencedError: If the image is not geo-referenced.
        :raises MapperMissingError: If no mapper is available.
        :raises CRSInputError: If the mapper rejects CRS/transformer input.
        :raises CoordinateTransformationError: If coordinate transformation fails.
        :raises WeitsichtError: Base class for all weitsicht exceptions. Catch this to handle any weitsicht error;
            catch specific subclasses first if you need to distinguish causes.
        """

        if not self.is_geo_referenced:
            raise NotGeoreferencedError()

        if self.mapper is None and mapper is None:
            raise MapperMissingError("No mapper provided for map_points")

        mapper_to_use = self.mapper
        if mapper is not None:
            mapper_to_use = mapper

        assert mapper_to_use is not None
        assert self.geo_transform is not None

        point_center_2d = self.geo_transform * [self.width / 2.0, self.height / 2.0]
        mapping_result = mapper_to_use.map_heights_from_coordinates(
            point_center_2d, None if transformer is not None else self._crs, transformer=transformer
        )

        if mapping_result.ok is False:
            return ResultFailure(ok=False, error=mapping_result.error, issues=mapping_result.issues)

        return MappingResultSuccess(
            ok=True,
            coordinates=mapping_result.coordinates,
            mask=mapping_result.mask,
            gsd=self.resolution,
            issues=mapping_result.issues,
        )

    def map_footprint(
        self,
        points_per_edge: int = 0,
        mapper: MappingBase | None = None,
        transformer: CoordinateTransformer | None = None,
    ) -> MappingResult:
        """Map the orthophoto footprint polygon to 3D coordinates via a mapper.

        :param points_per_edge: Number of points inserted between corners, defaults to ``0``.
            Values ``< 1`` map only the corner points.
        :type points_per_edge: int
        :param mapper: Mapper to use, defaults to ``None`` (uses :attr:`~weitsicht.image.base_class.ImageBase.mapper`).
        :type mapper: MappingBase | None
        :param transformer: Optional coordinate transformer, defaults to ``None``.
        :type transformer: CoordinateTransformer | None
        :return: Mapping result.
        :rtype: MappingResult
        :raises NotGeoreferencedError: If the image is not geo-referenced.
        :raises MapperMissingError: If no mapper is available.
        :raises CRSInputError: If the mapper rejects CRS/transformer input.
        :raises CoordinateTransformationError: If coordinate transformation fails.
        :raises WeitsichtError: Base class for all weitsicht exceptions. Catch this to handle any weitsicht error;
            catch specific subclasses first if you need to distinguish causes.
        """

        if not self.is_geo_referenced:
            raise NotGeoreferencedError()

        if self.mapper is None and mapper is None:
            raise MapperMissingError("No mapper provided for map_points")

        mapper_to_use = self.mapper
        if mapper is not None:
            mapper_to_use = mapper

        assert self.geo_transform is not None
        assert mapper_to_use is not None

        if points_per_edge < 1:
            px = np.array([0.0, 0.0, float(self.width), float(self.width)], dtype=np.float64)
            py = np.array([0.0, float(self.height), float(self.height), 0.0], dtype=np.float64)
        else:
            footprint_points_2d = [[x, 0] for x in np.linspace(0, self.width, 2 + points_per_edge)]
            footprint_points_2d += [[self.width, x] for x in np.linspace(0, self.height, 2 + points_per_edge)][1:]
            footprint_points_2d += [[x, self.height] for x in np.linspace(self.width, 0, 2 + points_per_edge)][1:]
            footprint_points_2d += [[0, x] for x in np.linspace(self.height, 0, 2 + points_per_edge)][1:-1]
            _fp = np.array(footprint_points_2d)
            px = _fp[:, 0]
            py = _fp[:, 1]

        pt_x, pt_y = cast(tuple[np.ndarray, np.ndarray], self.geo_transform * (px, py))
        footprint_points_2d = np.vstack((pt_x, pt_y)).T
        mapping_result = mapper_to_use.map_heights_from_coordinates(
            footprint_points_2d, None if transformer is not None else self._crs, transformer=transformer
        )

        if mapping_result.ok is False:
            return ResultFailure(ok=False, error=mapping_result.error, issues=mapping_result.issues)

        if not np.all(mapping_result.mask):
            return ResultFailure(
                ok=False,
                error="Some footprint points failed",
                issues=mapping_result.issues,
            )

        # TODO Implementation of resolution and area could be a problem for non cartesian georef of ortho
        footprint_geom = geometry.Polygon(mapping_result.coordinates)
        area = float(np.round(footprint_geom.area))
        return MappingResultSuccess(
            ok=True,
            coordinates=mapping_result.coordinates,
            mask=mapping_result.mask,
            gsd=self.resolution,
            area=area,
            issues=mapping_result.issues,
        )

    def map_points(
        self,
        points_image: ArrayNx2 | ArrayNx3 | list[list[float]] | list[list[int]] | list[float] | list[int],
        mapper: MappingBase | None = None,
        transformer: CoordinateTransformer | None = None,
    ) -> MappingResult:
        """Map orthophoto pixel coordinates to 3D coordinates via a mapper.

        :param points_image: Pixel coordinates (image CRS).
        :type points_image: ArrayNx2 | ArrayNx3 | list[list[float]] | list[list[int]] | list[float] | list[int]
        :param mapper: Mapper to use, defaults to ``None`` (uses :attr:`~weitsicht.image.base_class.ImageBase.mapper`).
        :type mapper: MappingBase | None
        :param transformer: Optional coordinate transformer, defaults to ``None``.
        :type transformer: CoordinateTransformer | None
        :return: Mapping result.
        :rtype: MappingResult
        :raises ValueError: If ``points_image`` cannot be parsed as an array of 2D points.
        :raises TypeError: If ``points_image`` has incompatible types.
        :raises NotGeoreferencedError: If the image is not geo-referenced.
        :raises MapperMissingError: If no mapper is available.
        :raises CRSInputError: If the mapper rejects CRS/transformer input.
        :raises CoordinateTransformationError: If coordinate transformation fails.
        :raises WeitsichtError: Base class for all weitsicht exceptions. Catch this to handle any weitsicht error;
            catch specific subclasses first if you need to distinguish causes.
        """

        # TODO what should we do if specified points_image are outside the orthophoto
        # but the mapping still works? Should be masked them as invalid?

        if not self.is_geo_referenced:
            raise NotGeoreferencedError()

        if self.mapper is None and mapper is None:
            raise MapperMissingError("No mapper provided for map_points")

        mapper_to_use = self.mapper
        if mapper is not None:
            mapper_to_use = mapper

        assert mapper_to_use is not None
        assert self.geo_transform is not None

        _points_image = to_array_nx2(points_image)

        pt_x, pt_y = cast(tuple[np.ndarray, np.ndarray], self.geo_transform * (_points_image[:, 0], _points_image[:, 1]))

        mapping_result = mapper_to_use.map_heights_from_coordinates(
            np.vstack((pt_x, pt_y)).T, None if transformer is not None else self._crs, transformer=transformer
        )

        if mapping_result.ok is False:
            return ResultFailure(ok=False, error=mapping_result.error, issues=mapping_result.issues)

        # TODO Implementation of resolution and area could be a problem for non cartesian georef of ortho
        return MappingResultSuccess(
            ok=True,
            coordinates=mapping_result.coordinates,
            mask=mapping_result.mask,
            gsd=self.resolution,
            issues=mapping_result.issues,
        )
