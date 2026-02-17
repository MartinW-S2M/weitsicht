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

"""Mapping backend based on an in-memory geo-referenced numpy array."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from affine import Affine
from pyproj import CRS

from weitsicht.exceptions import CRSInputError, CRSnoZaxisError
from weitsicht.geometry.interpolation_bilinear import bilinear_interpolation
from weitsicht.geometry.intersection_bilinear import multilinear_poly_intersection
from weitsicht.geometry.line_grid_intersection import (
    line_grid_intersection_points,
    raster_index_p1_p2,
    vector_projection,
)
from weitsicht.mapping.base_class import MappingBase, MappingType
from weitsicht.transform.coordinates_transformer import CoordinateTransformer
from weitsicht.utils import (
    ArrayN_,
    ArrayNx2,
    ArrayNx3,
    ArrayNxN,
    Issue,
    MappingResult,
    MappingResultSuccess,
    MaskN_,
    ResultFailure,
    to_array_nx3,
)

__all__ = ["MappingGeorefArray"]

logger = logging.getLogger(__name__)

neighbour_offsets = [
    np.array([0, 0]),
    np.array([1, -1]),
    np.array([1, 0]),
    np.array([1, 1]),
    np.array([0, 1]),
    np.array([0, -1]),
    np.array([-1, -1]),
    np.array([-1, 0]),
    np.array([-1, 1]),
]


# def tracer_bilinear_patch(points, pos_ray, ray):
#     bilinear_patch = BilinearPatch(*points)
#
#     ray_vec = Vector.from_vec(ray)
#     ray_vec.normalize()
# cover
#     pos = Vector.from_vec(pos_ray)
#     uv = bilinear_patch.ray_patch_intersection(pos_ray=pos,
#                                                ray=ray_vec)
#
#     if uv is not None:
#         return bilinear_patch.srf_eval(uv.x, uv.y).vec
#
#     return None


class MappingGeorefArray(MappingBase):
    """Mapping backend using an in-memory geo-referenced raster array.

    This backend is typically used internally (e.g. for raster window caching) and supports
    height sampling and ray intersections using bilinear patches.
    """

    def __init__(self, raster_array: ArrayNxN, geo_transform: Affine, crs: CRS | None):
        """Create a mapper backed by a numpy array and an affine geo-transform.

        :param raster_array: Raster data array (rows × cols) holding height values.
        :type raster_array: ArrayNxN
        :param geo_transform: Affine transformation mapping pixel indices to raster CRS coordinates.
        :type geo_transform: Affine
        :param crs: CRS of the raster array, defaults to ``None``.
        :type crs: CRS | None
        :raises CRSnoZaxisError: If a specified CRS does not define a Z axis.
        """
        super().__init__()

        self._raster_array = raster_array
        self._raster_array.setflags(write=False)
        self._transform: Affine = geo_transform
        self.area_flag = True

        # User CRS can override CRS from dataset
        self._crs = crs

        if self._crs is not None:
            if len(self._crs.axis_info) < 3:
                logger.error("CRS has no Z axis defined")
                raise CRSnoZaxisError("CRS has no Z axis defined")

        self._width = self._raster_array.shape[1]
        self._height = self._raster_array.shape[0]

    @property
    def transform(self):
        """Return the affine geo-transform.

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

    @classmethod
    def from_dict(cls, mapper_dict: Mapping[str, Any]) -> MappingGeorefArray:
        """Create a mapper from a configuration dictionary.

        This is currently not supported for :class:`MappingGeorefArray`.

        :raises NotImplementedError: Always raised.
        """
        raise NotImplementedError("This is not implemented for GeorefArray")

    @property
    def type(self) -> MappingType:
        """Return the mapper type."""
        return MappingType.GeoreferencedNumpyArray

    @property
    def param_dict(self) -> dict:
        """Return the mapper configuration dictionary.

        This is currently not supported for :class:`MappingGeorefArray`.

        :raises NotImplementedError: Always raised.
        """
        raise NotImplementedError("This is not implemented for GeorefArray")

    def pixel_to_coordinate(self, px_row: ArrayN_, px_col: ArrayN_) -> tuple:
        """Convert pixel indices to coordinates in raster CRS.

        :param px_row: Pixel row coordinate
        :type px_row: ``ArrayN_``
        :param px_col: Pixel column coordinate
        :type px_col: ``ArrayN_``
        :return: ``(x, y)`` coordinates in raster CRS.
        :rtype: tuple[``ArrayN_``, ``ArrayN_``]
        """
        return self.transform * np.vstack((px_col, px_row))

    def coordinate_to_pixel(self, x: ArrayN_, y: ArrayN_) -> tuple[ArrayN_, ArrayN_]:
        """Convert raster CRS coordinates to pixel indices.

        :param x: X coordinate in raster CRS.
        :type x: ``ArrayN_``
        :param y: Y coordinate in raster CRS.
        :type y: ``ArrayN_``
        :return: ``(row, col)`` pixel coordinates.
        :rtype: tuple[``ArrayN_``, ``ArrayN_``]
        """
        pix_col, pix_row = ~self.transform * np.vstack((x, y))
        return np.asarray(pix_row), np.asarray(pix_col)

    def pixel_valid(self, px_row: ArrayN_, px_col: ArrayN_) -> MaskN_:
        """Vectorized validity check for pixel coordinates.

        :param px_row: Pixel row coordinate
        :type px_row: ``ArrayN_``
        :param px_col: Pixel column coordinate
        :type px_col: ``ArrayN_``
        :return: Boolean mask for valid pixels.
        :rtype: ``MaskN_``
        """
        # 0    1    2    3    4
        # |----|----|----|----|
        #                   x
        # self.height = 100
        # px_row index can be from 0 to 99.9999

        # 0 <= px_row < self.height and 0 <= px_col < self.width
        # return bool(np.all((0 <= px_row, px_row < self.height, 0 <= px_col, px_col < self.width)))
        valid: MaskN_ = (px_row >= 0) & (px_row < self.height) & (px_col >= 0) & (px_col < self.width)
        return valid

    def pixel_valid_index_ray_bilinear_shifted(self, px_row: ArrayN_, px_col: ArrayN_) -> np.ndarray:
        """Return indices of pixels valid for bilinear ray intersection.

        This check assumes pixel coordinates are already shifted by ``0.5``. Border pixels are excluded
        because bilinear patches require 4 surrounding corner points.

        :param px_row: Pixel row coordinate already shifted by 0.5
        :type px_row: ``ArrayN_``
        :param px_col: Pixel column coordinate already shifted by 0.5
        :type px_col: ``ArrayN_``
        :return: Indices of valid pixels.
        :rtype: numpy.ndarray
        """
        # 0    1    2    3    4
        # |----|----|----|----|
        #                    x
        # self.height = 100
        # px_row index can be from 0.5 to 99.5
        # for the bilinear ray intersection the border region
        # is not valid as there the points are not within 4 points

        # here we say smaller than that because otherwise if a ray is vertical
        # and on the exact upper boundary there is a problem
        # bool_row = (px_row < self.height - 1) & (px_row >= 0)
        # bool_col = (px_col < self.width - 1) & (px_col >= 0)
        # index_combined = np.logical_and(bool_row, bool_col)
        return np.flatnonzero((px_row < self.height - 1) & (px_row >= 0) & (px_col < self.width - 1) & (px_col >= 0))

    def coordinate_on_raster(self, x_crs: ArrayN_, y_crs: ArrayN_) -> bool:
        """Test if given coordinates in raster CRS are on the raster

        :param x_crs: X coordinate
        :type x_crs: ``ArrayN_``
        :param y_crs: Y coordinate
        :type y_crs: ``ArrayN_``
        :return: ``True`` if all coordinates are within raster bounds.
        :rtype: bool
        """

        pix_row, pix_col = self.coordinate_to_pixel(x_crs, y_crs)
        return bool(np.all(self.pixel_valid(px_row=pix_row, px_col=pix_col)))

    def map_coordinates_from_rays_old_sampling(
        self, ray_vectors_crs_s: ArrayNx3, ray_start_crs_s: ArrayNx3, crs_s: CRS
    ) -> ArrayNx3 | None:  # pragma: no cover
        """Legacy ray intersection method using dense sampling (deprecated).

        :param ray_vectors_crs_s: Ray direction vectors (N×3).
        :type ray_vectors_crs_s: ArrayNx3
        :param ray_start_crs_s: Ray start points (N×3).
        :type ray_start_crs_s: ArrayNx3
        :param crs_s: CRS of the input rays.
        :type crs_s: CRS
        :return: Intersection coordinates or ``None`` if no intersection was found.
        :rtype: ArrayNx3 | None
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """

        # TODO The length and start of the sampling could be smart estimated, as well the sampling resolution
        # TODO we could as well make the length of the sampling vector in a for loop, so to extend distance

        coo_trafo = CoordinateTransformer.from_crs(crs_s=crs_s, crs_t=self._crs)

        intersection_point = []
        for idx_ray, ray in enumerate(ray_vectors_crs_s):
            # 1. Sample the ray vectors to form an array of points.
            # The sampling is not really of importance here as here only a
            # array is loaded which does not know how big the
            # source raster could be. And its only a numpy indexing. So we use 1cm.

            # TODO calc max distance possible for ray
            # max distance use minimum of height of georef array
            # intersection_plane(line_vec: np.ndarray, line_point: np.array([0,0,self._raster_array.min()]),
            # plane_point: np.ndarray, plane_normal: np.ndarray | None = None)

            dist_multi = np.arange(0, 900, 0.1)
            # This forms an array filled with unit vectors in the direction of the ray
            # which is than scaled by the sampled distance to get points along the ray
            ray_points_crs_s = (np.zeros((dist_multi.shape[0], 3)) + ray / np.linalg.norm(ray)).T * dist_multi
            # Now we translate the points to the correct location in 3d space
            ray_points_crs_s = ray_points_crs_s.T + ray_start_crs_s[idx_ray, :]

            # 2. Transform that points into the raster crs and check if all points are inside the raster
            if coo_trafo is not None:
                ray_points_crs_array = coo_trafo.transform(ray_points_crs_s)
            else:
                ray_points_crs_array = ray_points_crs_s * 1.0

            # Transform position of ray points in array crs to array col and row by the affine transformation
            # col, row = ~self._transform * ray_points_crs_array[:, :2].T
            row, col = self.coordinate_to_pixel(ray_points_crs_array[:, 0], ray_points_crs_array[:, 1])

            # 3. Find the first approx intersection of the ray and the raster

            # Only cells, where one of the corner points is higher and one lower as the sample point,
            # are potential intersection points
            # If all corner points are lower there can be no intersection in a bilinear patch

            # TODO check if outside raster? Or at least check if an intersection was found inside the raster
            # center_cells = np.hstack((np.floor(_row)+0.5, np.floor(_col)+0.5))
            # Find unique cells which can be tested. As sampling can lead to give the same cell more times
            # unique_rows = np.unique(center_cells, axis=0)

            # TODO: What would happen if one ray point is exactly on a raster edge or line?

            # TODO: A Problem is when using the same CRS and the ray is exactly on the line
            # Then for example if line vec = [0,0.3,-1] the col lower and col upper are equal
            # We should always get all surroundings cells not only via ceil and floor
            row_lower = np.floor(row).astype(int)
            row_upper = np.ceil(row).astype(int)
            col_lower = np.floor(col).astype(int)
            col_upper = np.ceil(col).astype(int)
            r1 = self._raster_array[row_lower, col_lower]
            r2 = self._raster_array[row_lower, col_upper]
            r3 = self._raster_array[row_upper, col_upper]
            r4 = self._raster_array[row_upper, col_lower]

            r_corners = np.vstack((r1, r2, r3, r4))

            cells_min = np.min(r_corners, axis=0)
            cells_max = np.max(r_corners, axis=0)

            # To find the indices where a possible intersection can occur,
            # 1) we check if a point is within the min an max of a cell and take this one and one before
            # Because the intersection could already happen before that point

            index_to_search: list[int] = []

            index_between: npt.NDArray[np.int_] = np.flatnonzero(
                (cells_max >= ray_points_crs_array[:, 2]) & (ray_points_crs_array[:, 2] >= cells_min)
            )

            if index_between.size != 0:
                index_to_search += [int(x) for x in index_between]
                # Add also the index before to it
                index_to_search += [int(x) - 1 for x in index_between]

            # 2) we test always 2 point of the ray if any of the raster values are within that 2 points

            stack_2_points = np.vstack((ray_points_crs_array[0:-1, 2], ray_points_crs_array[1:, 2]))
            stack_2_points_max = np.max(stack_2_points, axis=0)
            stack_2_points_min = np.max(stack_2_points, axis=0)

            stack_2_cells_max = np.vstack((cells_max[0:-1], cells_max[1:]))
            stack_2_cells_max_lower = np.min(stack_2_cells_max, axis=0)

            stack_2_cells_min = np.vstack((cells_max[0:-1], cells_max[1:]))
            stack_2_cells_min_higher = np.max(stack_2_cells_min, axis=0)

            index_between: npt.NDArray[np.int_] = np.flatnonzero(
                (stack_2_points_max >= stack_2_cells_max_lower) & (stack_2_points_min <= stack_2_cells_min_higher)
            )

            if index_between.size != 0:
                index_to_search += [int(x) for x in index_between]
                # Add also the index after should be added as we here get the index of the first cell
                # If we say that sampling is at least smaller than the raster sampling this is fine
                index_to_search += [int(x) + 1 for x in index_between]

            # Get unique index of cells to test with bilinear patches
            index_to_search = sorted(list(set(index_to_search)))

            # Now test all index starting from the first one till we have found an intersection
            # 4) Refine the intersection to be precise

            if len(index_to_search) > 0:
                # TODO: we could filter as well with numpy operation using r1 to r4
                #  instead of checking which cells are already looked. Should anyhow make not so much speed difference
                # We will only test cells which have not been tested. As the sampling can deliver same cells
                cells_already_looked = []

                # We now need to transform the four points into a cartesian space to run the bilinear intersection
                # TODO Maybe its better to transform all points of the for loop at once for runtime?

                uv = None
                for _index in index_to_search:
                    if uv is not None:
                        break
                    cell_corners_initial = np.array(
                        [
                            [row_lower[_index], col_lower[_index]],
                            [row_upper[_index], col_lower[_index]],
                            [row_lower[_index], col_upper[_index]],
                            [row_upper[_index], col_upper[_index]],
                        ]
                    )

                    for neighbour_offset in neighbour_offsets:
                        # The order of the points which create the bilinear patch is very important!!!
                        # the 4th point should be the one on the opposite side of point 1
                        cell_corners = cell_corners_initial + neighbour_offset

                        if cell_corners.tolist() in cells_already_looked:
                            continue

                        cells_already_looked.append(cell_corners.tolist())

                        # Get for corner pixel the coordinates in raster crs
                        x, y = self.pixel_to_coordinate(px_row=cell_corners[:, 0], px_col=cell_corners[:, 1])

                        # Create 3D point from pixel coordinates and raster value (height)
                        cell_3d_points = np.vstack(
                            (
                                x,
                                y,
                                self._raster_array[cell_corners[:, 0], cell_corners[:, 1]],
                            )
                        ).T

                        # Transform back to source crs.
                        if coo_trafo is not None:
                            patch_points_crs_s = coo_trafo.transform(cell_3d_points, direction="inverse")
                        else:
                            patch_points_crs_s = cell_3d_points * 1.0

                        # TODO Should we check if points coplanar?
                        # I think we would not gain any speed.

                        intersect = multilinear_poly_intersection(
                            patch_points_crs_s, p=ray_start_crs_s[idx_ray, :], r=ray
                        )

                        if intersect is not None:
                            intersection_point.append(intersect)
                            break

        if len(intersection_point) > 0:
            return np.array(intersection_point)
        return None

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

        _ray_vectors_crs_s = to_array_nx3(ray_vectors_crs_s)
        _ray_start_crs_s = to_array_nx3(ray_start_crs_s)

        if _ray_start_crs_s.shape != _ray_vectors_crs_s.shape:
            raise ValueError("Start of rays and rays have to be same size")

        # TODO The length and start of the sampling could be smart estimated, as well the sampling resolution
        # TODO we could as well make the length of the sampling vector in a for loop, so to extend distance

        if crs_s is not None and transformer is not None:
            raise CRSInputError("Either crs or transformation can be used or both None")
        coo_trafo = (
            transformer if transformer is not None else CoordinateTransformer.from_crs(crs_s=crs_s, crs_t=self._crs)
        )

        intersection_point = []
        valid_mask = []
        inside_mask = []
        for idx_ray, ray in enumerate(_ray_vectors_crs_s):
            # 1. Sample the ray vectors to get intersection lines of 50 meter.

            # TODO calc max distance possible for ray
            # max distance use minimum of height of georef array
            # intersection_plane(line_vec: np.ndarray, line_point: np.array([0,0,self._raster_array.min()]),

            # the maximum dist is for now 80km
            # Will walk in steps of 50 meter
            # we assume that within 50 meter any distortion by different projections will have not really an influence
            # will break iteration if the ray than is anyhow lower as we would then have already missed the intersection
            dist_multi = np.arange(0, 8000, 50)

            # This forms an array filled with unit vectors in the direction of the ray
            # which is than scaled by the sampled distance to get points along the ray
            ray_points_crs_s = (np.zeros((dist_multi.shape[0], 3)) + ray / np.linalg.norm(ray)).T * dist_multi

            del dist_multi
            # Now we translate the points to the correct location in 3d space
            ray_points_crs_s = ray_points_crs_s.T + _ray_start_crs_s[idx_ray, :]

            # 2. Transform that points into the raster crs
            # TODO what happens here if transformation grids are used. That would make the vector wrong
            # There can be height grids as well distortion grids on x, y -> This would be anyhow a big problem
            # Maybe we could only use height from starting points, if height grids are used this would work
            # Also if there would be rays with very long distance from start x,y to end x,y what happens with
            # mapping scales?
            if coo_trafo is not None:
                ray_points_crs_array = coo_trafo.transform(ray_points_crs_s)
            else:
                ray_points_crs_array = ray_points_crs_s * 1.0

            # Transform position of ray points in array crs to array col and row by the affine transformation
            pix_row, pix_col = self.coordinate_to_pixel(ray_points_crs_array[:, 0], ray_points_crs_array[:, 1])

            pix_row -= 0.5
            pix_col -= 0.5

            found_intersection_for_ray = False
            at_least_once_on_raster = False
            # Iterate now over the pieces of that ray
            for _idx in range(len(pix_row) - 1):
                # get the cells touching the line

                # if all raster z values of the cells are lower as the second point of the ray line
                # then go directly to the next line
                # TODO test both points. Like rays looking upward to mountain?

                x_1, x_2, y_1, y_2 = raster_index_p1_p2(
                    [pix_row[_idx], pix_col[_idx]],
                    [pix_row[_idx + 1], pix_col[_idx + 1]],
                )

                # TODO we should check if it was any time within the raster
                # otherwise we should return Issue.OUTSIDE_RASTER
                # The ray part is still outside the raster
                if (x_1 < 0 and x_2 < 0) or (x_1 > self.height - 1 and x_2 > self.height - 1):
                    continue

                if (y_1 < 0 and y_2 < 0) or (y_1 > self.width - 1 and y_2 > self.width - 1):
                    continue

                # As we here check the rectangle at which the projected part of the ray goes through for height
                # We need to make sare to be within the borders here as well..
                x_1 = 0 if x_1 < 0 else x_1
                y_1 = 0 if y_1 < 0 else y_1
                x_2 = 0 if x_2 < 0 else x_2
                y_2 = 0 if y_2 < 0 else y_2

                x_1 = self.height - 1 if x_1 > self.height - 1 else x_1
                y_1 = self.width - 1 if y_1 > self.width - 1 else y_1
                x_2 = self.height - 1 if x_2 > self.height - 1 else x_2
                y_2 = self.width - 1 if y_2 > self.width - 1 else y_2

                # Todo rays looking up?
                if not ((x_1 == x_2) or (y_1 == y_2)):
                    if np.max(self._raster_array[x_1 : x_2 + 1, y_1 : y_2 + 1]) < ray_points_crs_array[_idx + 1, 2]:
                        continue
                else:
                    if (x_1 == x_2) and (y_1 == y_2):
                        if self._raster_array[x_1, y_1] < ray_points_crs_array[_idx + 1, 2]:
                            continue
                    elif x_1 == x_2:
                        if np.max(self._raster_array[x_1, y_1 : y_2 + 1]) < ray_points_crs_array[_idx + 1, 2]:
                            continue
                    elif y_1 == y_2:
                        if np.max(self._raster_array[x_1 : x_2 + 1, y_1]) < ray_points_crs_array[_idx + 1, 2]:
                            continue

                cells = line_grid_intersection_points(
                    [pix_row[_idx], pix_col[_idx]],
                    [pix_row[_idx + 1], pix_col[_idx + 1]],
                )

                # In this case, differently to simple interpolation
                # by location we can not allow pixel positions within 0.5 of the border
                # In this area the interpolation is changed but this is not working for the rays

                # We filter only the valid ones
                index_valid = self.pixel_valid_index_ray_bilinear_shifted(cells[:, 0], cells[:, 1])

                if index_valid.size == 0:
                    continue

                # This is just a helper to check if for single rays it was at least once on the raster
                # So that we can return issue=OUTSIDE_RASTER
                at_least_once_on_raster = True

                cells = cells[index_valid, :]

                # Only cells, where corner points are higher and one lower as the sample point,
                # are potential intersection points
                # If all corner points are lower or higher there can be no intersection in a bilinear patch
                # if all corner points are higher we have already a problem because we missed the intersection

                # TODO check if outside raster? Or at least check if an intersection was found inside the raster

                cells_row1 = cells + np.array([1, 0])
                cells_col1 = cells + np.array([0, 1])
                cells_row1_col1 = cells + np.array([1, 1])

                r1 = self._raster_array[cells[:, 0], cells[:, 1]]
                r2 = self._raster_array[cells_row1[:, 0], cells_row1[:, 1]]
                r3 = self._raster_array[cells_col1[:, 0], cells_col1[:, 1]]
                r4 = self._raster_array[cells_row1_col1[:, 0], cells_row1_col1[:, 1]]
                r_corners = np.vstack((r1, r2, r3, r4))

                # if all raster z values of the cells are lower as the second point of the ray line
                # then go directly to the next line
                # this should actually already checked before
                if np.max(r_corners) < ray_points_crs_array[_idx + 1, 2]:
                    continue

                cells = np.hstack((cells, r1.reshape(-1, 1)))
                cells_row1 = np.hstack((cells_row1, r2.reshape(-1, 1)))
                cells_col1 = np.hstack((cells_col1, r3.reshape(-1, 1)))
                cells_row1_col1 = np.hstack((cells_row1_col1, r4.reshape(-1, 1)))

                cells_min = np.min(r_corners, axis=0)
                cells_max = np.max(r_corners, axis=0)

                # Another check to avoid checking of cells which can not be the solution
                cells_lower_than_ray = np.logical_and(
                    cells_min > ray_points_crs_array[_idx, 2],
                    cells_min > ray_points_crs_array[_idx + 1, 2],
                )

                cells_higher_than_ray = np.logical_and(
                    cells_max < ray_points_crs_array[_idx, 2],
                    cells_max < ray_points_crs_array[_idx + 1, 2],
                )

                cells_valid = np.flatnonzero(np.logical_not(np.logical_or(cells_lower_than_ray, cells_higher_than_ray)))

                if cells_valid.size == 0:
                    continue

                cells = cells[cells_valid, :]
                cells_row1 = cells_row1[cells_valid, :]
                cells_col1 = cells_col1[cells_valid, :]
                cells_row1_col1 = cells_row1_col1[cells_valid, :]

                r1 = None
                r2 = None
                r3 = None
                r4 = None

                # Now we will calculate all corner points height diff to the normal intersection point with the ray
                # Here we use the original pix_row and pix_col without the border changing
                p1 = np.array([pix_row[_idx], pix_col[_idx], ray_points_crs_array[_idx, 2]])
                p2 = np.array(
                    [
                        pix_row[_idx + 1],
                        pix_col[_idx + 1],
                        ray_points_crs_array[_idx + 1, 2],
                    ]
                )

                # intersection_plane_mat_operation(p1,p2-p1,cells)

                prj_1 = vector_projection(p1, p2, cells)
                prj_2 = vector_projection(p1, p2, cells_col1)
                prj_3 = vector_projection(p1, p2, cells_row1_col1)
                prj_4 = vector_projection(p1, p2, cells_row1)

                # This vertical stuff is not working for vertical lines
                # inside = np.logical_and(prj_1[:, 2] <= cells_max, prj_1[:, 2] >= cells_min)
                # inside = np.logical_or(inside, np.logical_and(prj_2[:, 2] <= cells_max, prj_2[:, 2] >= cells_min))
                # inside = np.logical_or(inside, np.logical_and(prj_3[:, 2] <= cells_max, prj_3[:, 2] >= cells_min))
                # inside = np.logical_or(inside, np.logical_and(prj_4[:, 2] <= cells_max, prj_4[:, 2] >= cells_min))

                inside = np.all(abs(prj_1[:, 0:2] - cells[:, 0:2]) <= 2, axis=1)
                inside = np.logical_and(inside, np.all(abs(prj_2[:, 0:2] - cells_col1[:, 0:2]) <= 2, axis=1))
                inside = np.logical_and(
                    inside,
                    np.all(abs(prj_3[:, 0:2] - cells_row1_col1[:, 0:2]) <= 2, axis=1),
                )
                inside = np.logical_and(inside, np.all(abs(prj_4[:, 0:2] - cells_row1[:, 0:2]) <= 2, axis=1))

                index_to_check = np.flatnonzero(inside)
                # Iteration over all the cell which the current piece of the ray is in 2d touched
                if len(index_to_check) > 0:
                    for _index in index_to_check:
                        # We run the bilinear intersection directly in pixel coordinates
                        # TODO Should we check if points coplanar?

                        # The order of the points which create the bilinear patch is very important!!!
                        # the 4th point should be the one on the opposite side of point 1
                        patch_points_crs_m = np.array(
                            [
                                cells[_index, :],
                                cells_col1[_index, :],
                                cells_row1[_index, :],
                                cells_row1_col1[_index, :],
                            ]
                        )

                        intersect = multilinear_poly_intersection(
                            patch_points_crs_m,
                            p=p1,
                            r=(p2 - p1) / np.linalg.norm(p2 - p1),
                        )

                        if intersect is not None:
                            intersect[0] += 0.5
                            intersect[1] += 0.5
                            intersection_point.append(intersect)
                            found_intersection_for_ray = True
                            break

                    if found_intersection_for_ray:
                        break

            if not found_intersection_for_ray:
                intersection_point.append(np.array([0, 0, 0]))

            valid_mask.append(found_intersection_for_ray)
            inside_mask.append(at_least_once_on_raster)

        intersection_point_array = np.array(intersection_point)
        if intersection_point_array.shape != ray_vectors_crs_s.shape:
            #    # Should be already handled in few lines above
            raise RuntimeError("*Intersection points are not the same size as ray vectors. This should not happen")

        if not any(inside_mask):
            return ResultFailure(
                ok=False,
                error="All was outside the raster",
                issues={Issue.OUTSIDE_RASTER},
            )

        mask = np.array(valid_mask, dtype=bool)
        mask_index = np.flatnonzero(mask)
        if not np.any(mask):
            return ResultFailure(
                ok=False,
                error="No intersections where found",
                issues={Issue.NO_INTERSECTION},
            )

        # We now transform everything back
        # first get world coordinates from pixel coordinates in mapper crs
        x, y = self.pixel_to_coordinate(
            px_row=intersection_point_array[mask_index, 0],
            px_col=intersection_point_array[mask_index, 1],
        )
        intersection_point_array[mask_index, 0] = x
        intersection_point_array[mask_index, 1] = y

        interp_source_crs = np.full(intersection_point_array.shape, np.nan, dtype=float)

        # Transform back to origin crs.
        if coo_trafo is not None:
            interp_source_crs[mask_index, :] = coo_trafo.transform(
                intersection_point_array[mask_index, :], direction="inverse"
            )
        else:
            interp_source_crs[mask_index, :] = np.array(intersection_point_array[mask_index, :]) * 1.0

        # TODO this should be just temporary, but transforming a normal vector from non cartesian coordinates
        # is even harder to validate and mostly then its not meter unit
        normals = np.full(intersection_point_array.shape, np.nan, dtype=float)
        normals_mapper_crs = np.array([0.0, 0.0, 1.0], dtype=float)
        if coo_trafo is not None:
            if mask_index.size > 0:
                normals[mask_index, :] = coo_trafo.transform_vector(
                    intersection_point_array[mask_index, :],
                    normals_mapper_crs,
                    direction="inverse",
                )
        else:
            normals[mask_index, :] = normals_mapper_crs

        issue = set()
        if not np.all(mask):
            issue.add(Issue.NO_INTERSECTION)
        if not all(inside_mask):
            issue.add(Issue.OUTSIDE_RASTER)
        return MappingResultSuccess(ok=True, coordinates=interp_source_crs, mask=mask, normals=normals, issues=issue)

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

        # Pure Coordinate Transformation. No Transformation of directions involved
        # 1) Transform coordinates if needed
        # 2) get index from transformed coordinates
        # 3) check if all indices are valid within the array limits
        # 4) get array values for valid index
        # 5) bilinear interpolation in pixel crs
        # 6) Assign estimated height to mapper crs points
        # 6) transform 3d points back

        _coordinates_crs_s = to_array_nx3(coordinates_crs_s, 0.0)

        # Transformer
        if crs_s is not None and transformer is not None:
            raise CRSInputError("Either crs or transformation can be used or both None")
        coo_trafo = (
            transformer if transformer is not None else CoordinateTransformer.from_crs(crs_s=crs_s, crs_t=self._crs)
        )

        if coo_trafo is not None:
            coo_mapper_crs = coo_trafo.transform(_coordinates_crs_s)
        else:
            coo_mapper_crs = _coordinates_crs_s * 1.0

        pixel_row, pixel_col = self.coordinate_to_pixel(x=coo_mapper_crs[:, 0], y=coo_mapper_crs[:, 1])

        # Check if all pixels are valid
        # Even if the original pixel location is 0.3,0.3 -> anyhow a bilinear transformation is not possible because
        # Than the four points would be [-0.5,-0.5],[0.5,0.5],[-0.5,0.5],[0.5,-0.5] which is not possible
        # So the shifted pixel is fine to be not valid
        valid_mask = self.pixel_valid(px_row=pixel_row, px_col=pixel_col)
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

        # As the value of the raster point is basically defined for the center
        # we shift our target location by 0.5 for the bilinear interpolation and
        # say raster values are defined for the integer raster location
        # that we do not have to deal finding the right cells involved. By shifting by 0.5 the involved cells are
        # the ones we get from floor and ceil, if we say the values are defined not in the center but in the edge points

        pixel_row -= 0.5
        pixel_col -= 0.5

        # edge cases height= 100 -> max row index for array values = 99
        # width=200, max col = 199 -> max column index for array values
        # lower than 0.0 can no pixel be as we're then already raising an Error
        # row 0.1 ; col 100.2 ->  0,1; 100,101
        # row 98.9 ; col 100.2 ->  98,99; 100,101
        # -> row 99 ; col 100.2 ->  98,99; 100,101 need to check if floor(row/col) is height-1/width-1

        # Gdal does treats for pixels from 0 to 0.5 just as 0.5
        pixel_row[pixel_row < 0] = 0.0
        pixel_col[pixel_col < 0] = 0.0

        pixel_row[pixel_row > (self.height - 1)] = self.height - 1
        pixel_col[pixel_col > (self.width - 1)] = self.width - 1

        # TODO Could there be numerical problems? for example if its -0.0000000001
        # Should we treat that as still valid?
        row_lower = np.floor(pixel_row).astype(int)
        row_upper = row_lower * 1
        row_upper[row_lower < self.height - 1] += 1
        # if row_upper is max pixel size then row_upper is that already
        # Bigger than self.height is not possible because than it would already not valid
        row_lower[row_lower == self.height - 1] -= 1

        col_lower = np.floor(pixel_col).astype(int)
        col_upper = col_lower * 1
        col_upper[col_lower < self.width - 1] += 1
        # if row_upper is max pixel size then row_upper is that already
        col_lower[col_lower == self.width - 1] -= 1

        # value of that pixels
        r1 = self._raster_array[row_lower, col_lower]
        r2 = self._raster_array[row_upper, col_lower]
        r3 = self._raster_array[row_lower, col_upper]
        r4 = self._raster_array[row_upper, col_upper]

        # bilinear Interpolation for all points in the pixel system of the array
        # Actually we could shift the pixel to zero, but I think the numeric range of
        # pixel dimension (grid size) is small enough to not run into numeric problems.

        for _index, _ in enumerate(row_lower):
            cell_corners = [
                [row_lower[_index], col_lower[_index], r1[_index]],
                [row_upper[_index], col_lower[_index], r2[_index]],
                [row_lower[_index], col_upper[_index], r3[_index]],
                [row_upper[_index], col_upper[_index], r4[_index]],
            ]

            value, _normal = bilinear_interpolation(points=cell_corners, x=pixel_row[_index], y=pixel_col[_index])

            coo_mapper_crs[valid_index[_index], 2] = value

        coo_source_crs = np.empty(coo_mapper_crs.shape, dtype=float)
        coo_source_crs.fill(np.nan)
        # Transform back to source crs.
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
