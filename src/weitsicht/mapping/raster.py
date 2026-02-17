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

"""Mapping backend based on raster height data (e.g. DEM) via :mod:`rasterio`."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyproj.exceptions
import rasterio
from affine import Affine
from pyproj import CRS
from rasterio import RasterioIOError
from rasterio.errors import CRSError
from rasterio.windows import Window
from shapely import box

from weitsicht.exceptions import CRSInputError, CRSnoZaxisError, MappingError
from weitsicht.geometry.interpolation_bilinear import bilinear_interpolation
from weitsicht.geometry.intersection_plane import intersection_plane
from weitsicht.mapping.base_class import MappingBase, MappingType
from weitsicht.mapping.georef_array import MappingGeorefArray
from weitsicht.transform.coordinates_transformer import CoordinateTransformer
from weitsicht.utils import (
    ArrayN_,
    ArrayNx2,
    ArrayNx3,
    Issue,
    MappingResult,
    MappingResultSuccess,
    MaskN_,
    ResultFailure,
    Vector3D,
    to_array_nx3,
)

__all__ = ["MappingRaster"]

logger = logging.getLogger(__name__)


class MappingRaster(MappingBase):
    """Mapping backend using a geo-referenced raster for height sampling and ray intersections."""

    # TODO Test rasters which are mirrored or flipped by geotransform

    def __init__(
        self,
        raster_path: Path | str,
        crs: CRS | None = None,
        force_no_crs: bool = False,
        preload_full_raster: bool = False,
        preload_window: None | tuple[float, float, float, float] = None,
        index_band: int | None = None,
    ):
        """Create a raster-based mapper.

        Provide a valid path for a raster which can be loaded via :mod:`rasterio`.

        :param raster_path: Path to a raster readable by :mod:`rasterio`.
        :type raster_path: Path | str
        :param crs: CRS overriding the raster's internal CRS, defaults to ``None``.
        :type crs: CRS | None
        :param force_no_crs: If ``True`` the CRS will be set to ``None`` regardless of raster metadata,
            defaults to ``False``.
        :type force_no_crs: bool
        :param preload_full_raster: If ``True`` preload the full raster into memory, defaults to ``False``.
        :type preload_full_raster: bool
        :param preload_window: Optional window to preload as ``(xmin, ymin, xmax, ymax)`` in mapper CRS,
            defaults to ``None``.
        :type preload_window: tuple[float, float, float, float] | None
        :param index_band: Band index to load (1-based). Mandatory if raster has more than one band,
            defaults to ``None``.
        :type index_band: int | None
        :raises ValueError: If input arguments are inconsistent or invalid.
        :raises FileNotFoundError: If ``raster_path`` does not exist.
        :raises CRSInputError: If CRS input is missing or invalid.
        :raises CRSnoZaxisError: If the mapper CRS does not define a Z axis.
        :raises MappingError: If the raster cannot be opened or is not geo-referenced.
        """

        super().__init__()

        if preload_full_raster and preload_window is not None:
            raise ValueError("Preload Full raster and Preload window can not be used together")

        if preload_window:
            if len(preload_window) != 4:
                raise ValueError(
                    f"Preload window needs to be a tuple or list with 4 items but {len(preload_window)} are given"
                )

        if force_no_crs and crs is not None:
            raise ValueError("force_no_crs can not be used together with crs argument")

        self.path = Path(raster_path)

        if not self.path.exists():
            raise FileNotFoundError(f"File specified not Found: {self.path.as_posix()}")

        try:
            self._dataset: rasterio.DatasetReader = rasterio.open(self.path)
        except RasterioIOError as err:
            raise MappingError("File format not working for rasterio") from err

        if self._dataset.count > 1 and index_band is None:
            raise ValueError("Dataset has more than 1 band but no band was specified. Band numbers start with 1")

        if index_band is not None and index_band > self._dataset.count:
            raise ValueError("Band index was specified but is higher than number of bands in raster file")

        if index_band is not None:
            self.index_band = index_band
        else:
            self.index_band = 1

        # get affine geo transform
        self._transform: Affine = self._dataset.transform
        if self._transform.is_identity:
            raise MappingError("Geo Transform of raster is Identity. Probably missing or no geo-referenced raster")

        # RasterIO reads images in array with row is the first index if seen like a numpy array
        self._height = self._dataset.shape[0]
        self._width = self._dataset.shape[1]

        self._georef_array: MappingGeorefArray | None = None

        # TODO this should be optimized
        if self._dataset.crs is not None:
            try:
                linear_unit_factor = self._dataset.crs.linear_units_factor
            except CRSError:
                linear_unit_factor = [1.0, 1.0]
            self._res = np.mean(self._dataset.res) * linear_unit_factor[1]

        else:
            self._res = np.mean(self._dataset.res)

        # Force crs will leave crs None
        if force_no_crs:
            self._crs = None
        else:
            # get crs code from raster
            if self._dataset.crs is not None and crs is None:
                try:
                    self._crs: CRS | None = CRS(self._dataset.crs)
                except pyproj.exceptions.CRSError as err:
                    raise MappingError("Crs inside file is not working") from err

            # User CRS can override CRS from dataset

            if crs is not None:
                self._crs = crs

            if self._crs is None:
                raise CRSInputError("No CRS was specified or missing in file")

            if self._crs is not None:
                if len(self._crs.axis_info) < 3:
                    logger.error("CRS has no Z axis defined")
                    raise CRSnoZaxisError("CRS has no Z axis defined")

        self.preload_full_raster = preload_full_raster
        if preload_full_raster:
            self.load_window()

        self.preload_window = preload_window
        if preload_window is not None:
            self.load_window(preload_window)

    @property
    def type(self):
        return MappingType.Raster

    @property
    def backend(self) -> MappingBase:
        """Return the active mapping backend.

        If a raster window (or the full raster) was preloaded via :meth:`load_window` /
        ``preload_window`` / ``preload_full_raster``, this returns the in-memory
        :class:`~weitsicht.mapping.georef_array.MappingGeorefArray` backend. Otherwise,
        it returns ``self``.
        """
        return self._georef_array if self._georef_array is not None else self

    @property
    def georef_mapper(self) -> MappingGeorefArray:
        """Return the in-memory :class:`~weitsicht.mapping.georef_array.MappingGeorefArray` backend.

        :raises MappingError: If no in-memory window/full raster was loaded.
        """
        if self._georef_array is None:
            raise MappingError(
                "georef_array is not loaded. Call load_window(...) or set preload_window / preload_full_raster."
            )
        return self._georef_array

    @classmethod
    def from_dict(cls, mapper_dict: Mapping[str, Any]) -> MappingRaster:
        """Create a :class:`MappingRaster` instance from a configuration dictionary.

        Required keys:
        - ``raster_filepath``: file path to the raster
        - ``crs``: CRS WKT string (or ``None`` to force no CRS)

        Optional keys:
        - ``band``: band index (1-based), defaults to ``None``
        - ``preload_window``: preload window in mapper CRS as ``(xmin, ymin, xmax, ymax)``, defaults to ``None``
        - ``preload_full``: preload full raster, defaults to ``False``

        :param mapper_dict: Mapper configuration dictionary (typically created via ``mapper.param_dict``).
        :type mapper_dict: Mapping[str, Any]
        :return: Instantiated mapper.
        :rtype: MappingRaster
        :raises KeyError: If a required dictionary key is missing.
        :raises FileNotFoundError: If the raster file does not exist.
        :raises ValueError: If configuration values are invalid.
        :raises CRSInputError: If the CRS WKT string is invalid.
        :raises CRSnoZaxisError: If the mapper CRS does not define a Z axis.
        :raises MappingError: If the raster cannot be opened or is not geo-referenced.
        """

        try:
            raster_path = Path(mapper_dict["raster_filepath"])
        except KeyError as err:
            raise KeyError("Dictionary key 'raster_filepath' is missing") from err

        try:
            crs_text = mapper_dict["crs"]
        except KeyError as err:
            raise KeyError("Dictionary key 'crs' is missing") from err

        if crs_text is not None:
            try:
                crs = CRS(crs_text)
            except pyproj.exceptions.CRSError as err:
                raise CRSInputError("No valid CRS wkt string supported") from err
        else:
            crs = None
        force_no_crs = True if crs is None else False

        index_band = mapper_dict.get("band", None)
        preload_window = mapper_dict.get("preload_window", None)
        preload_full = mapper_dict.get("preload_full", False)
        mapping = cls(
            raster_path=raster_path,
            crs=crs,
            force_no_crs=force_no_crs,
            preload_full_raster=preload_full,
            preload_window=preload_window,
            index_band=index_band,
        )
        return mapping

    @property
    def param_dict(self):
        """Return the mapper configuration dictionary.

        :return: Mapper configuration dictionary.
        :rtype: dict[str, Any]
        """
        return_dict = {
            "type": self.type.fullname,
            "raster_filepath": self.path.as_posix(),
            "crs": self.crs_wkt,
            "band": self.index_band,
        }

        if self.preload_full_raster:
            return_dict.update({"preload_full": True})
        if self.preload_window is not None:
            return_dict.update({"preload_window": self.preload_window})
        return return_dict

    @property
    def resolution(self):
        """Return the raster resolution.

        :return: Mean raster resolution in CRS units.
        :rtype: float
        """
        return self._res

    @property
    def transform(self) -> Affine:
        """Return the affine geo-transform of the raster.

        :return: Affine geo-transform.
        :rtype: Affine
        """
        return self._transform

    @property
    def width(self):
        """Return raster width in pixels."""
        return self._width

    @property
    def height(self):
        """Return raster height in pixels."""
        return self._height

    def load_window(self, limits_crs: tuple[float, float, float, float] | None = None) -> MappingGeorefArray:
        """Load raster data into :attr:`georef_array`.

        :param limits_crs: Optional window limits as ``(xmin, ymin, xmax, ymax)`` in mapper CRS,
            defaults to ``None`` (loads full raster).
        :type limits_crs: tuple[float, float, float, float] | None
        :raises MappingError: If the provided window is not fully inside the raster extent.
        """

        # TODO this is a cached operation read()
        # So it could be smart to load bigger window and reuse that raster mapper
        # Then the reading of the georef_array is without IO
        # https://rasterio.readthedocs.io/en/stable/api/rasterio.io.html

        if limits_crs is None:
            raster_array = self._dataset.read(indexes=self.index_band)
            self._georef_array = MappingGeorefArray(
                raster_array=raster_array, geo_transform=self.transform, crs=self._crs
            )
            return self._georef_array

        # Test if limits inside window
        raster_bounds = self._dataset.bounds
        shape_raster_bounds = box(*raster_bounds)
        shape_limits = box(*limits_crs)

        if not shape_raster_bounds.contains(shape_limits):
            raise MappingError("Provided window is not fully inside raster's extend")

        x_min, y_min, x_max, y_max = limits_crs

        win = self._dataset.window(x_min, y_min, x_max, y_max)

        # As the limit are within the raster extend rounding of the pixel window should be not of a problem
        # as then row and col can only be >= 0.0
        # For numerical reasons (not sure if this can ever happen) we round to 0 everything lower than 1.0
        win_col_offset_rounded = np.floor(win.col_off)
        if win.col_off < 0:
            win_col_offset_rounded = 0
        win_row_offset_rounded = np.floor(win.row_off)
        if win.row_off < 0:
            win_row_offset_rounded = 0

        col_max = np.ceil(win.col_off + win.width)
        row_max = np.ceil(win.row_off + win.height)
        width = col_max - win_col_offset_rounded + 1
        height = row_max - win_row_offset_rounded + 1
        rounded_win = Window(
            col_off=win_col_offset_rounded,  # type: ignore
            row_off=win_row_offset_rounded,  # type: ignore
            width=width,  # type: ignore
            height=height,  # type: ignore
        )
        affine_of_window = self._dataset.window_transform(rounded_win)

        self._georef_array = MappingGeorefArray(
            raster_array=self._dataset.read(indexes=self.index_band, window=rounded_win),
            geo_transform=affine_of_window,
            crs=self._crs,
        )
        return self._georef_array

    def pixel_to_coordinate(self, px_row: float, px_col: float) -> tuple:
        """Convert pixel indices to coordinates in raster CRS.

        :param px_row: Pixel row coordinate
        :type px_row: float
        :param px_col: Pixel column coordinate
        :type px_col: float
        :return: ``(x, y)`` coordinate in raster CRS.
        :rtype: tuple[float, float]
        """
        return self.transform * (px_col, px_row)

    def pixel_valid(self, px_row: float, px_col: float) -> bool:
        """Return whether pixel coordinates are within raster bounds.

        :param px_row: Pixel row coordinate
        :type px_row: float
        :param px_col: Pixel column coordinate
        :type px_col: float
        :return: ``True`` if valid.
        :rtype: bool
        """
        return 0 <= px_row < self.height and 0 <= px_col < self.width

    def pixel_valid_mat(self, px_row: ArrayN_, px_col: ArrayN_) -> MaskN_:
        """Vectorized validity check for pixel coordinates.

        :param px_row: Array with Pixel row coordinate
        :type px_row: ``ArrayN_``
        :param px_col: Array with Pixel column coordinate
        :type px_col: ``ArrayN_``
        :return: Boolean mask for valid pixels.
        :rtype: ``MaskN_``
        """
        # 0 <= px_row < self.height and 0 <= px_col < self.width
        return (0 <= px_row) & (px_row < self.height) & (0 <= px_col) & (px_col < self.width)

    def coordinate_on_raster(self, x_crs: float, y_crs: float) -> bool:
        """Test if given coordinates in raster CRS are on the raster

        :param x_crs: X coordinate
        :type x_crs: float
        :param y_crs: Y coordinate
        :type y_crs: float
        :return: ``True`` if on the raster.
        :rtype: bool
        """

        pix_row, pix_col = self.coordinate_to_pixel(x_crs, y_crs)
        return self.pixel_valid(px_row=pix_row, px_col=pix_col)

    def coordinate_to_pixel(self, x_crs: float, y_crs: float) -> tuple[float, float]:
        """Get pixel position of coordinates

        :param x_crs: X coordinate
        :type x_crs: float
        :param y_crs: Y coordinate
        :type y_crs: float
        :return: ``(row, col)`` pixel coordinates as floats.
        :rtype: tuple[float, float]
        """
        gt = cast(Affine, ~self.transform)
        pix_col, pix_row = cast(tuple[float, float], gt * (x_crs, y_crs))

        return pix_row, pix_col

    def coordinate_to_pixel_mat(self, x: ArrayN_, y: ArrayN_) -> tuple[ArrayN_, ArrayN_]:
        """Get pixel position of coordinates.

        This is pure affine transformation, it will not check if the pixel are valid

        :param x: Array of X coordinate
        :type x: ``ArrayN_``
        :param y: Array of Y coordinate
        :type y: ``ArrayN_``
        :return: ``(row, col)`` pixel coordinates.
        :rtype: tuple[``ArrayN_``, ``ArrayN_``]
        """

        pix_col, pix_row = ~self.transform * np.vstack((x, y))
        return pix_row, pix_col

    def get_coordinate_height(self, x_crs: float, y_crs: float) -> float | None:
        """Get raster value of coordinates in Raster CRS coordinates

        :param x_crs: X coordinate
        :type x_crs: float
        :param y_crs: Y coordinate
        :type y_crs: float
        :return: Raster value or ``0.0`` if no-data / invalid.
        :rtype: float
        """
        # TODO also here if raster coordinates are valid?
        # Its only called internally but if user wants to call that by themselfs
        height = None
        for val in self._dataset.sample([(x_crs, y_crs)], indexes=self.index_band):
            height = val[0]
            if height in self._dataset.get_nodatavals():
                # if math.isclose(height, self._dataset.nodata):
                return None

        return height

    def intersection_ray(
        self,
        ray_vector_crs_s: Vector3D,
        ray_start_crs_s: Vector3D,
        crs_s: CRS | None = None,
        transformer: CoordinateTransformer | None = None,
        max_iter: int = 20,
    ) -> tuple[Vector3D, bool, bool, bool, Vector3D]:
        """Calculate the intersection of a single ray with the raster surface.

        The algorithm iteratively adjusts the intersection plane using heights sampled from the raster.

        :param ray_vector_crs_s: Ray direction vector (3,).
        :type ray_vector_crs_s: Vector3D
        :param ray_start_crs_s: Ray start point (3,).
        :type ray_start_crs_s: Vector3D
        :param crs_s: CRS of the ray input, defaults to ``None``.
        :type crs_s: CRS | None
        :param transformer: Coordinate transformer to mapper CRS, defaults to ``None``.
        :type transformer: CoordinateTransformer | None
        :param max_iter: Maximum number of iterations, defaults to ``20``.
        :type max_iter: int
        :return: Tuple ``(intersection_coordinates, is_invalid, is_outside_raster, Nodata height, normal in crs_s)``.
        :rtype: tuple[Vector3D, bool, bool, Vector3D]
        :raises CRSInputError: If both ``crs_s`` and ``transformer`` are provided.
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """

        intersect_coo = np.array([0, 0, 0])

        if crs_s is not None and transformer is not None:
            raise CRSInputError("Either crs or transformation can be used or both None")
        coo_trafo = (
            transformer if transformer is not None else CoordinateTransformer.from_crs(crs_s=crs_s, crs_t=self._crs)
        )

        if coo_trafo is not None:
            plane_point_mapper_crs = coo_trafo.transform(ray_start_crs_s)[0, :]
        else:
            plane_point_mapper_crs = ray_start_crs_s * 1.0

        # initial height is height from image position
        height = self.get_coordinate_height(x_crs=plane_point_mapper_crs[0], y_crs=plane_point_mapper_crs[1])

        if height is None:
            return intersect_coo, False, False, True, np.array([np.nan, np.nan, np.nan])

        # height for the initial itteration we will set as vertical half way from ray start and raster height.
        height_iteration = plane_point_mapper_crs[2] / 2.0 + height / 2.0

        _p = np.array([plane_point_mapper_crs[0], plane_point_mapper_crs[1], height_iteration])
        if coo_trafo is not None:
            plane_point_source_crs = coo_trafo.transform(_p, direction="inverse")[0, :]
            plane_normal_source_crs = coo_trafo.transform_vector(_p, np.array([0, 0, 1]), direction="inverse")[0, :]

        else:
            plane_point_source_crs = _p
            plane_normal_source_crs = np.array([0, 0, 1])

        count_iteration = 0

        intersect_coo = np.array([0, 0, 0])

        norm_diff = np.linalg.norm(plane_point_source_crs - ray_start_crs_s)
        # Iterate till the difference is smaller than 0.02meter or 40 iteations are over
        # We check if the ray distance between itterations is less than 1mm
        while (norm_diff > 0.02 and count_iteration < 40) or count_iteration < 1:
            intersect_coo_old = intersect_coo.copy()
            # get the intersection point
            intersect_coo, valid_intersection = intersection_plane(
                ray_vector_crs_s, ray_start_crs_s, plane_point_source_crs, plane_normal_source_crs
            )
            norm_diff = np.linalg.norm(intersect_coo - intersect_coo_old)

            # check if the direction is not backwards
            if not valid_intersection:
                return intersect_coo, True, False, False, np.array([np.nan, np.nan, np.nan])

            # transform back from source crs in 3d space to mapper crs to get new height
            # We will transform the intersection point and grap the raster height on this positions.
            # As well we are transforming the local vertical normal in the mapper crs at that point to the source crs.

            # Transform to mapper crs
            if coo_trafo is not None:
                plane_point_mapper_crs = coo_trafo.transform(
                    np.array([intersect_coo[0], intersect_coo[1], intersect_coo[2]])
                )[0, :]
            else:
                plane_point_mapper_crs = np.array([intersect_coo[0], intersect_coo[1], intersect_coo[2]])

            # Check if we are still on the raster, otherwise return OUTSIDE flag
            if not self.coordinate_on_raster(plane_point_mapper_crs[0], plane_point_mapper_crs[1]):
                return intersect_coo, False, True, False, np.array([np.nan, np.nan, np.nan])

            # Get height from raster and return NO_DATA_TOUCHED if not data was the raster cell'S value
            height_pixel = self.get_coordinate_height(plane_point_mapper_crs[0], plane_point_mapper_crs[1])
            count_iteration += 1
            if height_pixel is None:
                return intersect_coo, False, False, True, np.array([np.nan, np.nan, np.nan])

            # This is to be sure that we will not run in a loop
            height_iteration = height_iteration * 0.02 + height_pixel * 0.98

            # Now transform the new point back (with the updated height) to source crs as well the verical normal
            _p = np.array([plane_point_mapper_crs[0], plane_point_mapper_crs[1], height_iteration])
            if coo_trafo is not None:
                plane_point_source_crs = coo_trafo.transform(
                    _p,
                    direction="inverse",
                )[0, :]
                plane_normal_source_crs = coo_trafo.transform_vector(_p, np.array([0, 0, 1]), direction="inverse")[0, :]
            else:
                plane_point_source_crs = _p
                plane_normal_source_crs = np.array([0, 0, 1])

            # That was not correct as here we would shit the wrong intersection point
            # intersect_coo = np.array([intersect_coo[0], intersect_coo[1], height_transformed])

        if count_iteration >= max_iter:
            return intersect_coo, True, True, True, plane_normal_source_crs

        return intersect_coo, False, False, False, plane_normal_source_crs

    def map_coordinates_from_rays(
        self,
        ray_vectors_crs_s: ArrayNx3,
        ray_start_crs_s: ArrayNx3,
        crs_s: CRS | None = None,
        transformer: CoordinateTransformer | None = None,
    ) -> MappingResult:
        """Map coordinates by intersecting rays with the raster surface.

        :param ray_vectors_crs_s: Ray direction vectors (N×3).
        :type ray_vectors_crs_s: ArrayNx3
        :param ray_start_crs_s: Ray start points (N×3), same shape as ``ray_vectors_crs_s``.
        :type ray_start_crs_s: ArrayNx3
        :param crs_s: CRS of the input rays, defaults to ``None``.
        :type crs_s: CRS | None
        :param transformer: Coordinate transformer to mapper CRS, defaults to ``None``.
        :type transformer: CoordinateTransformer | None
        :return: Mapping result containing intersection coordinates and a validity mask.
        :rtype: MappingResult
        :raises ValueError: If input arrays have incompatible shapes.
        :raises CRSInputError: If both ``crs_s`` and ``transformer`` are provided.
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """

        # We will transform the coordinate of the raster crs with the height taken from the raster
        # to the image crs for intersection
        # So that we do not need to transform the ray vector according to the projection
        # It is for sure slower than transforming the ray vector one time but this is easier to implement
        # As well with that approach the raster can be geographic as well without the need to translate a raster subset

        _ray_vectors_crs_s = to_array_nx3(ray_vectors_crs_s)
        _ray_start_crs_s = to_array_nx3(ray_start_crs_s)

        if _ray_start_crs_s.shape != _ray_vectors_crs_s.shape:
            raise ValueError("Start of rays and rays have to be same size")

        if crs_s is not None and transformer is not None:
            raise CRSInputError("Either crs or transformation can be used or both None")

        if self._georef_array is not None:
            return self._georef_array.map_coordinates_from_rays(
                ray_vectors_crs_s=_ray_vectors_crs_s,
                ray_start_crs_s=_ray_start_crs_s,
                transformer=transformer,
                crs_s=crs_s,
            )

        list_coo = []
        valid = []
        normals = []
        issue = set()

        for idx, coordinates in enumerate(_ray_start_crs_s):
            res = self.intersection_ray(_ray_vectors_crs_s[idx, :], coordinates, transformer=transformer, crs_s=crs_s)
            inter_p, invalid_flag, outside_flag, nodata_touched, normal = res

            normals.append(normal)

            if outside_flag:
                inter_p = np.array([np.nan, np.nan, np.nan])
                issue.add(Issue.OUTSIDE_RASTER)
                # raise MappingError("Intersection is outside of provided raster")
                # return MappingResultFailure(ok=False, error=["Intersection is outside of provided raster"],
                # issues=issue)
            if invalid_flag:
                inter_p = np.array([np.nan, np.nan, np.nan])
                issue.add(Issue.WRONG_DIRECTION)
            if nodata_touched:
                inter_p = np.array([np.nan, np.nan, np.nan])
                issue.add(Issue.RASTER_NO_DATA)
            if outside_flag and invalid_flag and nodata_touched:
                inter_p = np.array([np.nan, np.nan, np.nan])
                issue.add(Issue.MAX_ITTERATION)

            valid.append(not invalid_flag and not outside_flag and not nodata_touched)
            list_coo.append(inter_p)

        coo_crs_source = np.array(list_coo)
        normals_crs_source = np.array(normals)
        mask = np.array(valid, dtype=bool)

        if not np.all(mask):
            if not np.any(mask):
                return ResultFailure(
                    ok=False,
                    error="None of the rays could be intersected",
                    issues=issue,
                )

        return MappingResultSuccess(
            ok=True,
            coordinates=coo_crs_source,
            mask=mask,
            normals=normals_crs_source,
            issues=issue,
            crs=crs_s,
        )

    def map_heights_from_coordinates_sampling(
        self,
        coordinates_crs_s: ArrayNx3 | ArrayNx2,
        crs_s: CRS | None = None,
        transformer: CoordinateTransformer | None = None,
    ) -> MappingResult:  # pragma: no cover
        """Sample heights using rasterio's point sampling (no interpolation).

        :param coordinates_crs_s: Coordinates to sample (N×2 or N×3).
        :type coordinates_crs_s: ArrayNx3 | ArrayNx2
        :param crs_s: CRS of the input coordinates, defaults to ``None``.
        :type crs_s: CRS | None
        :param transformer: Coordinate transformer to mapper CRS, defaults to ``None``.
        :type transformer: CoordinateTransformer | None
        :return: Mapping result containing sampled coordinates and a validity mask.
        :rtype: MappingResult
        :raises CRSInputError: If both ``crs_s`` and ``transformer`` are provided.
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """

        coordinates_crs_s = to_array_nx3(coordinates_crs_s)

        # Convert Nx2 array to Nx3
        if coordinates_crs_s.shape[1] < 3:
            coordinates_crs_s = np.hstack((coordinates_crs_s, np.zeros((coordinates_crs_s.shape[0], 1))))

        if crs_s is not None and transformer is not None:
            raise CRSInputError("Either crs or transformation can be used or both None")
        coo_trafo = (
            transformer if transformer is not None else CoordinateTransformer.from_crs(crs_s=crs_s, crs_t=self._crs)
        )

        if coo_trafo is not None:
            coo_mapper_crs = coo_trafo.transform(coordinates_crs_s)
        else:
            coo_mapper_crs = coordinates_crs_s * 1.0

        # 2)
        pixel_row, pixel_col = self.coordinate_to_pixel_mat(x=coo_mapper_crs[:, 0], y=coo_mapper_crs[:, 1])

        # 3)
        valid_mask = self.pixel_valid_mat(px_row=pixel_row, px_col=pixel_col)
        issue = set()
        if not np.all(valid_mask):
            issue = {Issue.OUTSIDE_RASTER}
            if not np.any(valid_mask):
                return ResultFailure(
                    ok=False,
                    error="None of the coordinates are on the raster",
                    issues={Issue.OUTSIDE_RASTER},
                )

        height = np.fromiter(
            (val[0] for val in self._dataset.sample(coo_mapper_crs[valid_mask, :2], indexes=self.index_band)),
            dtype=float,
        )

        coo_mapper_crs[valid_mask, 2] = height
        coo_source_crs = np.full(coordinates_crs_s.shape, np.nan, dtype=float)
        # Transform back to source crs.
        if coo_trafo is not None:
            coo_source_crs[valid_mask, :] = coo_trafo.transform(coo_mapper_crs[valid_mask, :], direction="inverse")
        else:
            coo_source_crs[valid_mask, :] = coo_mapper_crs[valid_mask, :] * 1.0

        normals = np.full(coo_source_crs.shape, np.nan, dtype=float)
        normals_mapper_crs = np.array([0.0, 0.0, 1.0], dtype=float)
        if coo_trafo is not None:
            mask_index = np.flatnonzero(valid_mask)
            if mask_index.size > 0:
                normals[mask_index, :] = coo_trafo.transform_vector(
                    coo_mapper_crs[mask_index, :],
                    normals_mapper_crs,
                    direction="inverse",
                )
        else:
            normals[valid_mask, :] = normals_mapper_crs

        return MappingResultSuccess(
            ok=True,
            coordinates=coo_source_crs,
            mask=valid_mask,
            normals=normals,
            crs=crs_s,
            issues=issue,
        )

    def map_heights_from_coordinates(
        self,
        coordinates_crs_s: ArrayNx3 | ArrayNx2,
        crs_s: CRS | None = None,
        transformer: CoordinateTransformer | None = None,
    ) -> MappingResult:
        """Sample heights for given coordinates using bilinear interpolation.

        :param coordinates_crs_s: Coordinates to sample (N×2 or N×3).
        :type coordinates_crs_s: ArrayNx3 | ArrayNx2
        :param crs_s: CRS of the input coordinates, defaults to ``None``.
        :type crs_s: CRS | None
        :param transformer: Coordinate transformer to mapper CRS, defaults to ``None``.
        :type transformer: CoordinateTransformer | None
        :return: Mapping result containing sampled coordinates and a validity mask.
        :rtype: MappingResult
        :raises CRSInputError: If both ``crs_s`` and ``transformer`` are provided.
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """

        # Only here we do bilinear interpolation
        # For that mapper on the rays we just use the cell value

        # Convert Nx2 array to Nx3 and fill with zeros if needed
        _coordinates_crs_s = to_array_nx3(coordinates_crs_s, fill_z=0.0)
        # TODO converting 2D to 3D by 0.0 could be of a problem if a 3d projection is used (like crs_s is ECEF)
        # Some projections are only accurate if z coordinate is present. Not sure if any raster will be in that CRS
        # Adding 0.0 to Array2N as z will then cause problems.
        # Maybe add in docu if 2D coordinates are used here that they will be filled up
        # Normally no one would provide 2D coordinates for ECEF for example

        if crs_s is not None and transformer is not None:
            raise CRSInputError("Either crs or transformation can be used or both None")

        if self._georef_array is not None:
            return self._georef_array.map_heights_from_coordinates(
                coordinates_crs_s=_coordinates_crs_s,
                crs_s=crs_s,
                transformer=transformer,
            )
        coo_trafo = (
            transformer if transformer is not None else CoordinateTransformer.from_crs(crs_s=crs_s, crs_t=self._crs)
        )

        if coo_trafo is not None:
            coo_mapper_crs = coo_trafo.transform(_coordinates_crs_s)
        else:
            coo_mapper_crs = _coordinates_crs_s * 1.0

        pixel_row, pixel_col = self.coordinate_to_pixel_mat(x=coo_mapper_crs[:, 0], y=coo_mapper_crs[:, 1])

        valid_mask = self.pixel_valid_mat(px_row=pixel_row, px_col=pixel_col)

        valid_index = np.flatnonzero(valid_mask)
        issue = set()
        if not np.all(valid_mask):
            issue = {Issue.OUTSIDE_RASTER}
            if not np.any(valid_mask):
                return ResultFailure(
                    ok=False,
                    error="None of the coordinates are on the raster",
                    issues={Issue.OUTSIDE_RASTER},
                )

        # use only valid pixels
        pixel_row = pixel_row[valid_mask]
        pixel_col = pixel_col[valid_mask]

        # Shift as height of raster cell is defined for center point
        pixel_row = pixel_row - 0.5
        pixel_col = pixel_col - 0.5

        # Gdal does treat pixels from 0 to 0.5 just as 0.5
        pixel_row[pixel_row < 0] = 0.0
        pixel_col[pixel_col < 0] = 0.0

        pixel_row[pixel_row > (self.height - 1)] = self.height - 1
        pixel_col[pixel_col > (self.width - 1)] = self.width - 1

        row_lower = np.floor(pixel_row).astype(int)
        row_upper = row_lower * 1
        row_upper[row_lower < self.height - 1] += 1
        # if row_upper is max pixel size then row_upper is that already
        row_lower[row_lower == self.height - 1] -= 1

        col_lower = np.floor(pixel_col).astype(int)
        col_upper = col_lower * 1
        col_upper[col_lower < self.width - 1] += 1
        # if row_upper is max pixel size then row_upper is that already
        col_lower[col_lower == self.width - 1] -= 1

        for _index, row in enumerate(row_lower):
            win = Window(col_off=col_lower[_index], row_off=row, width=2, height=2)  # type: ignore
            data = self._dataset.read(indexes=self.index_band, window=win)

            cell_corners = [
                [row_lower[_index], col_lower[_index], data[0, 0]],
                [row_upper[_index], col_lower[_index], data[1, 0]],
                [row_lower[_index], col_upper[_index], data[0, 1]],
                [row_upper[_index], col_upper[_index], data[1, 1]],
            ]

            value, _normal = bilinear_interpolation(points=cell_corners, x=pixel_row[_index], y=pixel_col[_index])

            # We could use very well .read( with resampling=bliniear) here
            # win = Window(col_off=pixel_col[_index], row_off=pixel_row[_index], width=1, height=1)
            # data = self._dataset.read(1, window=win, resampling=rasterio.enums.Resampling.bilinear)
            # But resampled read is not cached so this may decrease performance

            coo_mapper_crs[valid_index[_index], 2] = value

        # Transform back to source crs.
        coo_source_crs = np.empty(coo_mapper_crs.shape, dtype=float)
        coo_source_crs.fill(np.nan)
        if coo_trafo is not None:
            coo_source_crs[valid_index, :] = coo_trafo.transform(coo_mapper_crs[valid_index, :], direction="inverse")
        else:
            coo_source_crs[valid_index, :] = coo_mapper_crs[valid_index, :] * 1.0

        normals = np.full(coo_source_crs.shape, np.nan, dtype=float)
        normals_mapper_crs = np.array([0.0, 0.0, 1.0], dtype=float)
        if coo_trafo is not None:
            if valid_index.size > 0:
                normals[valid_index, :] = coo_trafo.transform_vector(
                    coo_mapper_crs[valid_index, :],
                    normals_mapper_crs,
                    direction="inverse",
                )
        else:
            normals[valid_index, :] = normals_mapper_crs

        return MappingResultSuccess(
            ok=True,
            coordinates=coo_source_crs,
            mask=valid_mask,
            normals=normals,
            crs=crs_s,
            issues=issue,
        )
