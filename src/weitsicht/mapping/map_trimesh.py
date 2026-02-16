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

"""Mapping backend based on a :mod:`trimesh` mesh surface."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyproj.exceptions
import trimesh
from pyproj import CRS

from weitsicht.exceptions import CRSInputError, CRSnoZaxisError, MappingError
from weitsicht.geometry.ray_from_points import rays_from_points_batch
from weitsicht.mapping.base_class import MappingBase, MappingType
from weitsicht.transform.coordinates_transformer import CoordinateTransformer
from weitsicht.utils import (
    ArrayNx2,
    ArrayNx3,
    Issue,
    MappingResult,
    MappingResultSuccess,
    ResultFailure,
    Vector3D,
    to_array_nx3,
)

__all__ = ["MappingTrimesh"]

logger = logging.getLogger(__name__)
# attach to logger so trimesh messages will be printed to console
# trimesh.util.attach_to_log()


class MappingTrimesh(MappingBase):
    """Mapping backend that intersects rays with a mesh surface."""

    def __init__(
        self,
        mesh_path: Path | str,
        crs: CRS | None = None,
        coordinate_shift: Vector3D | None = None,
    ):
        """Create a mesh-based mapper.

        :param mesh_path: Path to the mesh file.
        :type mesh_path: Path | str
        :param crs: CRS of the mesh, defaults to ``None``.
        :type crs: CRS | None
        :param coordinate_shift: Optional translation applied to the mesh coordinates, defaults to ``None``.
        :type coordinate_shift: Vector3D | None
        :raises FileNotFoundError: If ``mesh_path`` does not exist.
        :raises CRSnoZaxisError: If a specified CRS does not define a Z axis.
        :raises MappingError: If the mesh is empty or the file format is unsupported.
        """
        super().__init__()

        self.path = Path(mesh_path)

        self.translation = coordinate_shift

        if not self.path.exists():
            raise FileNotFoundError(f"File specified not Found: {self.path.as_posix()}")

        try:
            # load_mesh needs Pillow to be installed, I was not able to shut that off
            # Thats the reason we are casting and checking that self._mnesh is a trimesh.Trimesh object
            # trimesh.load() returns GEOMETRY -> supercalls. but geometry has no ray function ->problem for linting
            self._mesh = trimesh.load(self.path)  # pyright: ignore[reportUnknownMemberType]
            assert isinstance(self._mesh, trimesh.Trimesh)
        except KeyError as e:
            raise MappingError(f"File format {self.path.suffix} not working for trimesh") from e
        except NotImplementedError as e:
            raise MappingError(f"File format {self.path.suffix} not working for trimesh") from e
        except ValueError as e:
            raise MappingError("File seems not working") from e

        if self._mesh.is_empty:
            raise MappingError("Trimesh object is empty")

        self._crs = crs

        if self._crs is not None:
            if len(self._crs.axis_info) < 3:
                logger.error("CRS has no Z axis defined")
                raise CRSnoZaxisError("CRS has no Z axis defined")

    @property
    def type(self):
        """Return the mapper type."""
        return MappingType.Trimesh

    @classmethod
    def from_dict(cls, mapper_dict: Mapping[str, Any]) -> MappingTrimesh:
        """Create a :class:`MappingTrimesh` instance from a configuration dictionary.

        Required keys:
        - ``type``: must be ``Trimesh``
        - ``mesh_filepath``: file path to the mesh

        Optional keys:
        - ``crs``: CRS WKT string, defaults to ``None``
        - ``coordinate_shift``: translation vector, defaults to ``None``

        :param mapper_dict: Mapper configuration dictionary (typically created via ``mapper.param_dict``).
        :type mapper_dict: Mapping[str, Any]
        :return: Instantiated mapper.
        :rtype: MappingTrimesh
        :raises KeyError: If a required dictionary key is missing.
        :raises ValueError: If the mapper type is unsupported.
        :raises CRSInputError: If the CRS WKT string is invalid.
        """

        try:
            type_text = mapper_dict["type"]
        except KeyError as err:
            raise KeyError("'type' key is missing") from err

        if type_text != MappingType.Trimesh.fullname:
            raise ValueError(f'"type" of mapper dict is not {MappingType.Trimesh.fullname}')

        crs_text = mapper_dict.get("crs", None)

        try:
            mesh_path = Path(mapper_dict["mesh_filepath"])
        except KeyError as err:
            raise KeyError('"mesh_filepath" key is missing') from err

        coordinate_shift = None
        if mapper_dict.get("coordinate_shift", None) is not None:
            coordinate_shift = np.array(mapper_dict["coordinate_shift"])

        crs = None

        # Get CRS if specified
        if crs_text is not None:
            try:
                crs = CRS(crs_text)
            except pyproj.exceptions.CRSError as err:
                raise CRSInputError("No valid CRS wkt string supported") from err

        mapping = cls(mesh_path=mesh_path, crs=crs, coordinate_shift=coordinate_shift)

        return mapping

    @property
    def param_dict(self):
        """Return the mapper configuration dictionary.

        :return: Mapper configuration dictionary.
        :rtype: dict[str, Any]
        """

        dict_return = {
            "type": self.type.fullname,
            "mesh_filepath": self.path.as_posix(),
            "crs": self.crs_wkt,
            "coordinate_shift": self.translation.tolist() if self.translation is not None else None,
        }
        return dict_return

    def map_coordinates_from_rays(
        self,
        ray_vectors_crs_s: ArrayNx3,
        ray_start_crs_s: ArrayNx3,
        crs_s: CRS | None = None,
        transformer: CoordinateTransformer | None = None,
    ) -> MappingResult:
        """Map coordinates by intersecting rays with the mesh surface.

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

        rays_vector = to_array_nx3(ray_vectors_crs_s)
        ray_start = to_array_nx3(ray_start_crs_s)

        if ray_start.shape != rays_vector.shape:
            raise ValueError("Array size for ray vectors and ray start points do not fit")

        if crs_s is not None and transformer is not None:
            raise CRSInputError("Either crs or transformation can be used or both None")
        coo_trafo = (
            transformer if transformer is not None else CoordinateTransformer.from_crs(crs_s=crs_s, crs_t=self._crs)
        )

        if coo_trafo is not None:
            dists = np.array([0, 50, 200, 700])
            points_batch = ray_start[:, None, :] + dists[None, :, None] * rays_vector[:, None, :]
            # ray_start_crs_m = coo_trafo.transform(ray_start)
            _ray_pts_crs_m = coo_trafo.transform(points_batch)
            _, rays_vector_crs_m = rays_from_points_batch(_ray_pts_crs_m, use_first_segment=True)
            ray_start_crs_m = _ray_pts_crs_m[:, 0, :]
        else:
            ray_start_crs_m = ray_start * 1.0
            rays_vector_crs_m = rays_vector * 1.0

        # Trimesh to find intersection locations
        # It returns the intersection location, the index of the ray and the index of the triangle of the mesh
        assert isinstance(self._mesh, trimesh.Trimesh)
        loc_crs_m_raw, index_ray, _ = self._mesh.ray.intersects_location(  # pyright: ignore[reportUnknownMemberType]
            ray_origins=ray_start_crs_m,
            ray_directions=rays_vector_crs_m,
            multiple_hits=False,
        )

        loc_crs_m_raw: ArrayNx3 = np.array(loc_crs_m_raw)
        if loc_crs_m_raw.size == 0:
            issue = {Issue.NO_INTERSECTION}
            return ResultFailure(ok=False, error="No intersections where found", issues=issue)

        # if index_ray.shape[0] != ray_start_crs_m.shape[0]:
        #    raise MappingError("Not for all rays, intersections were found")
        mask = np.zeros(ray_vectors_crs_s.shape[0], dtype=bool)
        mask[index_ray] = True

        issue = set()
        if not np.all(mask):
            issue = {Issue.NO_INTERSECTION}

        # Sort the resulting location array to have same order as input
        # Trimesh returns unordered array for locations - but index_ray is given
        # sort_index = np.argsort(index_ray)
        # locations_crs_m = loc_crs_m_raw[sort_index, :]

        # Transform back to source crs.
        if coo_trafo is not None:
            coo_crs_source = coo_trafo.transform(loc_crs_m_raw, direction="inverse")
        else:
            coo_crs_source = loc_crs_m_raw * 1.0

        coo_crs_result = np.empty(rays_vector.shape, dtype=float)
        coo_crs_result.fill(np.nan)
        coo_crs_result[index_ray, :] = coo_crs_source

        return MappingResultSuccess(ok=True, coordinates=coo_crs_result, mask=mask, crs=crs_s, issues=issue)

    def map_heights_from_coordinates(
        self,
        coordinates_crs_s: ArrayNx3 | ArrayNx2,
        crs_s: CRS | None = None,
        transformer: CoordinateTransformer | None = None,
    ) -> MappingResult:
        """Sample heights by casting vertical rays against the mesh.

        :param coordinates_crs_s: Coordinates to sample (N×2 or N×3).
        :type coordinates_crs_s: ArrayNx3 | ArrayNx2
        :param crs_s: CRS of the input coordinates, defaults to None.
        :type crs_s: CRS | None
        :param transformer: Coordinate transformer to mapper CRS, defaults to None.
        :type transformer: CoordinateTransformer | None
        :return: Mapping result containing sampled coordinates and a validity mask.
        :rtype: MappingResult
        :raises CRSInputError: If both ``crs_s`` and ``transformer`` are provided.
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """

        # originally I wanted to check for vertical ray in both direction (up, down)
        # and use the higher mesh intersection for both rays, but it could be that there are multiple hits down and up
        # Now I just change the coordinates to have a z component of 9000 as a temporary solution
        # TODO maybe make that vertical grapping more advanced see comment before

        coordinates_crs_s = to_array_nx3(coordinates_crs_s, 0.0)

        h = 9000.0
        coordinates_crs_s[:, 2] = h

        if crs_s is not None and transformer is not None:
            raise CRSInputError("Either crs or transformation can be used or both None")
        coo_trafo = (
            transformer if transformer is not None else CoordinateTransformer.from_crs(crs_s=crs_s, crs_t=self._crs)
        )

        if coo_trafo is not None:
            coordinates_mapper_crs = coo_trafo.transform(coordinates_crs_s)
        else:
            coordinates_mapper_crs = coordinates_crs_s * 1.0

        rays_vertical = np.zeros(coordinates_crs_s.shape)
        rays_vertical[:, 2] = -1

        result_mapper_crs = self.map_coordinates_from_rays(
            ray_vectors_crs_s=rays_vertical,
            ray_start_crs_s=coordinates_mapper_crs,
            crs_s=self._crs,
        )

        if result_mapper_crs.ok is False:
            return ResultFailure(ok=False, error=result_mapper_crs.error, issues=result_mapper_crs.issues)

        # Transform back to source crs.
        # Non valid coordinates will be nan
        coo_crs_source = np.empty(coordinates_mapper_crs.shape, dtype=float)
        coo_crs_source.fill(np.nan)
        index = np.flatnonzero(result_mapper_crs.mask)
        if coo_trafo is not None:
            coo_crs_source[index, :] = coo_trafo.transform(result_mapper_crs.coordinates[index, :], direction="inverse")
        else:
            coo_crs_source[index, :] = result_mapper_crs.coordinates[index, :] * 1.0

        return MappingResultSuccess(
            ok=True, coordinates=coo_crs_source, mask=result_mapper_crs.mask, crs=crs_s, issues=result_mapper_crs.issues
        )
