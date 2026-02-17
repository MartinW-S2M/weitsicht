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

"""Mapping backend for a horizontal plane at a fixed altitude."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pyproj.exceptions
from pyproj import CRS

from weitsicht.exceptions import CRSInputError
from weitsicht.geometry.intersection_plane import intersection_plane_mat_operation
from weitsicht.mapping.base_class import MappingBase, MappingType
from weitsicht.transform.coordinates_transformer import CoordinateTransformer
from weitsicht.utils import (
    ArrayNx2,
    ArrayNx3,
    Issue,
    MappingResult,
    MappingResultSuccess,
    ResultFailure,
    to_array_nx3,
)

__all__ = ["MappingHorizontalPlane"]

logger = logging.getLogger(__name__)


class MappingHorizontalPlane(MappingBase):
    """Mapping backend for a horizontal plane at a fixed altitude.

    The plane is defined by ``plane_altitude`` (Z coordinate in mapper CRS) and a fixed normal (0, 0, 1).
    If a CRS transformation is performed, the plane altitude is transformed appropriately via the provided
    ``crs_s``/``transformer`` inputs.
    """

    def __init__(self, plane_altitude: float | int = 0.0, crs: CRS | None = None):
        """Create a horizontal plane mapper.

        :param plane_altitude: Altitude of the horizontal plane (Z value in mapper CRS), defaults to ``0.0``.
        :type plane_altitude: float | int
        :param crs: CRS of the plane, defaults to ``None``.
        :type crs: CRS | None
        """

        super().__init__()

        self.plane_altitude = float(plane_altitude)
        self.plane_normal = np.array([0, 0, 1])
        self._crs = crs

        # TODO check if crs is not None that the CRS provides a vertical reference

    @classmethod
    def from_dict(cls, mapper_dict: Mapping[str, Any]) -> MappingHorizontalPlane:
        """Create a :class:`MappingHorizontalPlane` instance from a configuration dictionary.

        Required keys:
        - ``type``: must be ``horizontalPlane``

        Optional keys:
        - ``plane_altitude``: plane altitude (float), defaults to ``0.0``
        - ``crs``: CRS WKT string, defaults to ``None``

        :param mapper_dict: Mapper configuration dictionary (typically created via ``mapper.param_dict``).
        :type mapper_dict: Mapping[str, Any]
        :return: Instantiated mapper.
        :rtype: MappingHorizontalPlane
        :raises KeyError: If the dictionary key ``type`` is missing.
        :raises ValueError: If the mapper type is unsupported.
        :raises CRSInputError: If the CRS WKT string is invalid.
        """

        if mapper_dict.get("type", None) is None:
            raise KeyError("Dictionary key 'type' is missing")

        if mapper_dict.get("type", None) != MappingType.HorizontalPlane.fullname:
            raise ValueError("Mapper dictionary type is not " + MappingType.HorizontalPlane.fullname)

        plane_altitude = mapper_dict.get("plane_altitude", 0.0)
        crs_text = mapper_dict.get("crs", None)

        crs = None
        if crs_text is not None:
            try:
                crs = CRS(crs_text)
            except pyproj.exceptions.CRSError as err:
                raise CRSInputError("No valid CRS WKT string supported") from err

        mapping = cls(plane_altitude=plane_altitude, crs=crs)
        return mapping

    @property
    def type(self):
        """Return the mapper type."""
        return MappingType.HorizontalPlane

    @property
    def param_dict(self):
        """Return the mapper configuration dictionary.

        :return: Mapper configuration dictionary.
        :rtype: dict[str, Any]
        """

        crs_text = None
        if self._crs is not None:
            crs_text = self.crs_wkt

        return {
            "type": self.type.fullname,
            "plane_altitude": self.plane_altitude,
            "crs": crs_text,
        }

    def map_coordinates_from_rays(
        self,
        ray_vectors_crs_s: ArrayNx3,
        ray_start_crs_s: ArrayNx3,
        crs_s: CRS | None = None,
        transformer: CoordinateTransformer | None = None,
    ) -> MappingResult:
        """Map coordinates from rays to a horizontal plane.

        :param ray_vectors_crs_s: Ray direction vectors (N×3).
        :type ray_vectors_crs_s: ArrayNx3
        :param ray_start_crs_s: Ray start points (N×3), same shape as ``ray_vectors_crs_s``.
        :type ray_start_crs_s: ArrayNx3
        :param crs_s: CRS of the input rays, defaults to None.
        :type crs_s: CRS | None
        :param transformer: Coordinate transformer to mapper CRS, defaults to None.
        :type transformer: CoordinateTransformer | None
        :return: Mapping result containing intersection coordinates and a validity mask.
        :rtype: MappingResult
        :raises ValueError: If input arrays have incompatible shapes.
        :raises CRSInputError: If both ``crs_s`` and ``transformer`` are provided.
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """

        # A Array(3,) (Vector3D) should also work ray_vectors_crs_s and ray_start_crs_s
        ray_pos = to_array_nx3(ray_start_crs_s)
        ray_vec = to_array_nx3(ray_vectors_crs_s)

        if ray_pos.shape != ray_vec.shape:
            raise ValueError("Ray and Pos Vector not the same size")

        # transform the given coordinates to the mappers crs
        # replace the z coordinate of the coordinates in mapper crs with the mappers height
        # transform back to the source CRS which now provides mapper height in source crs
        # Therefore the mapper's height is correct transformed to the image crs at the image's prj center
        if crs_s is not None and transformer is not None:
            raise CRSInputError("Either crs or transformation can be used or both None")
        coo_trafo = (
            transformer if transformer is not None else CoordinateTransformer.from_crs(crs_s=crs_s, crs_t=self._crs)
        )

        # We will do this iterative, so if in the coordinate transformation raster are used
        # We will iterate the plane height to that one from the correct intersection point
        # Thus also valid mask should be taken into account, otherwise coordinate operation are wrong
        # If both coo_trafo is None we will skip iteration

        prev_intersect_points = np.ones(ray_pos.shape, dtype=float)
        diff_greater_1mm = np.ones(ray_pos.shape, dtype=float)
        iteration: int = 0
        valid_mask = np.ones((ray_pos.shape[0],), dtype=bool)
        plane_point_source_crs = np.ones(ray_pos.shape, dtype=float)
        intersect_points = np.empty(ray_pos.shape, dtype=float)
        intersect_points.fill(np.nan)

        # initial height will be taken from ray start position
        if coo_trafo is not None:
            plane_point_mapper_crs = coo_trafo.transform(ray_pos)
        else:
            plane_point_mapper_crs = ray_pos * 1.0

        plane_point_mapper_crs[:, 2] = self.plane_altitude

        while np.any(diff_greater_1mm > 0.001) and iteration < 20:
            index_valid_mask = np.flatnonzero(valid_mask)
            # Transform back to source crs. So we have the horizontal planes Altitude in the correct system
            if coo_trafo is not None:
                plane_point_source_crs[index_valid_mask, :] = coo_trafo.transform(
                    plane_point_mapper_crs[index_valid_mask, :], direction="inverse"
                )
            else:
                plane_point_source_crs[index_valid_mask, :] = plane_point_mapper_crs[index_valid_mask, :] * 1.0

            # _intersect_points, _valid_mask are corresponding to size of true items in valid_mask
            # its very unlikely that validity changes between iterations
            # only if the plane point would then be exactly the ray pos it would be invalid
            # so at least always grapping validity we can reduce risk here
            # Testing that case would be lot of work, creating a pipeline which transform exactly like this.
            _intersect_points, _valid_mask = intersection_plane_mat_operation(
                ray_vec[index_valid_mask, :],
                ray_pos[index_valid_mask, :],
                plane_point=plane_point_source_crs[index_valid_mask, :],
                plane_normal=self.plane_normal,
            )
            _index_false = np.flatnonzero(~_valid_mask)
            _index_correct = np.flatnonzero(_valid_mask)
            valid_mask[index_valid_mask[_index_false]] = False
            intersect_points[index_valid_mask[_index_correct], :] = _intersect_points[_index_correct, :]
            if coo_trafo is None:
                break
            if not np.any(valid_mask):
                break
            diff_greater_1mm = np.abs(intersect_points[valid_mask, :] - prev_intersect_points[valid_mask, :])
            prev_intersect_points = intersect_points * 1.0
            iteration += 1
            # Now the new plane point is the previous intersection point

            if coo_trafo is not None:
                plane_point_mapper_crs[valid_mask, :] = coo_trafo.transform(intersect_points[valid_mask, :])
                plane_point_mapper_crs[valid_mask, 2] = self.plane_altitude
            else:
                plane_point_mapper_crs[valid_mask, :] = ray_pos[valid_mask, :] * 1.0

        issue = set()
        if not np.all(valid_mask):
            issue = {Issue.NO_INTERSECTION}
            if not np.any(valid_mask):
                return ResultFailure(
                    ok=False,
                    error="None of the rays intersects",
                    issues={Issue.NO_INTERSECTION},
                )

        normals_mapper_crs = np.full(plane_point_mapper_crs.shape, np.nan, dtype=float)
        normals_mapper_crs[valid_mask, :] = np.array([0, 0, 1.0])

        normals = np.full(plane_point_mapper_crs.shape, np.nan, dtype=float)
        if coo_trafo is not None:
            normals[valid_mask, :] = coo_trafo.transform_vector(
                plane_point_mapper_crs[valid_mask, :],
                normals_mapper_crs[valid_mask, :],
                direction="inverse",
            )
        else:
            normals = normals_mapper_crs

        return MappingResultSuccess(
            ok=True,
            coordinates=intersect_points,
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
        """Assign the plane altitude as height for given coordinates.

        :param coordinates_crs_s: Coordinates to sample (N×2 or N×3).
        :type coordinates_crs_s: ArrayNx3 | ArrayNx2
        :param crs_s: CRS of the input coordinates, defaults to None.
        :type crs_s: CRS | None
        :param transformer: Coordinate transformer to mapper CRS, defaults to None.
        :type transformer: CoordinateTransformer | None
        :return: Mapping result containing coordinates with plane altitude and a validity mask.
        :rtype: MappingResult
        :raises CRSInputError: If both ``crs_s`` and ``transformer`` are provided.
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """

        # Prepare input if something else than ArrayNx3 (e.g. List, ArrayN)
        coordinates = to_array_nx3(coordinates_crs_s, 0.0)

        if crs_s is not None and transformer is not None:
            raise CRSInputError("Either crs or transformation can be used or both None")
        coo_trafo = (
            transformer if transformer is not None else CoordinateTransformer.from_crs(crs_s=crs_s, crs_t=self._crs)
        )

        if coo_trafo is not None:
            coo_mapper_crs = coo_trafo.transform(coordinates)
        else:
            coo_mapper_crs = coordinates * 1.0

        # Now change the altitude to the plane height
        coo_mapper_crs[:, 2] = self.plane_altitude

        # Transform back to source crs. So we have the horizontal planes Altitude in the correct system
        # Here no iteration is needed.
        # Or at least if we not assume that the transformation with a new height will not change the 2d coordinates
        if coo_trafo is not None:
            coo_source_crs = coo_trafo.transform(coo_mapper_crs, direction="inverse")
        else:
            coo_source_crs = coo_mapper_crs * 1.0

        # In that case all are valid because its simple assigned its heights
        # Any coordinate error would have already raised an error before
        mask = np.ones((coo_source_crs.shape[0]), dtype=bool)

        normals_mapper_crs = np.full(coo_source_crs.shape, np.array([0, 0, 1]), dtype=float)

        if coo_trafo is not None:
            normals = coo_trafo.transform_vector(coo_mapper_crs, normals_mapper_crs, direction="inverse")
        else:
            normals = normals_mapper_crs

        return MappingResultSuccess(ok=True, coordinates=coo_source_crs, mask=mask, normals=normals, crs=crs_s)
