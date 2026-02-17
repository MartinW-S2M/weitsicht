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

"""Perspective image model backed by a camera perspective implementation."""

from __future__ import annotations

import logging

import numpy as np
import pyproj.exceptions
from pyproj import CRS
from shapely import geometry

from weitsicht.camera.base_perspective import CameraBasePerspective
from weitsicht.camera.camera_dict_selector import get_camera_from_dict
from weitsicht.exceptions import (
    CRSInputError,
    MapperMissingError,
    MappingBackendError,
    NotGeoreferencedError,
    WeitsichtError,
)
from weitsicht.geometry.intersection_plane import intersection_plane_mat_operation
from weitsicht.image.base_class import ImageBase, ImageType
from weitsicht.mapping.base_class import MappingBase
from weitsicht.transform.coordinates_transformer import CoordinateTransformer
from weitsicht.transform.rotation import Rotation
from weitsicht.utils import (
    ArrayN_,
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
    to_array_nx3,
)

__all__ = ["ImagePerspective"]

logger = logging.getLogger(__name__)


class ImagePerspective(ImageBase):
    """Perspective image model.

    The image is geo-referenced if position, orientation, camera and CRS are specified.
    The camera model implementation should support resampled images (i.e. account for
    calibration size vs. current image size).
    """

    def __init__(
        self,
        width: float | int,
        height: float | int,
        camera: CameraBasePerspective | None,
        crs: CRS | None = None,
        position: Vector3D | None = None,
        orientation: Rotation | None = None,
        mapper: MappingBase | None = None,
    ):
        """Initialize a perspective image model.

        :param width: Image width in pixels.
        :type width: float | int
        :param height: Image height in pixels.
        :type height: float | int
        :param camera: Camera model used for projection/distortion, may be ``None`` for non-geo-referenced instances.
        :type camera: CameraBasePerspective | None
        :param crs: World CRS of the image, defaults to ``None``.
        :type crs: CRS | None
        :param position: Camera position in world CRS, defaults to ``None``.
        :type position: Vector3D | None
        :param orientation: Camera orientation, defaults to ``None``.
        :type orientation: Rotation | None
        :param mapper: Mapping instance, defaults to ``None``.
        :type mapper: MappingBase | None
        """

        super().__init__(mapper=mapper, crs=crs, width=width, height=height)

        self._position: Vector3D | None = position
        self._orientation: Rotation | None = orientation

        self._camera = camera

        # TODO check if crs is not None that the CRS provides a vertical reference

    @property
    def type(self) -> ImageType:
        """Return the image model type.

        :return: Image type.
        :rtype: ImageType
        """
        return ImageType.Perspective

    @classmethod
    def from_dict(cls, param_dict: dict, mapper: MappingBase | None = None) -> ImagePerspective:
        """Create an :class:`ImagePerspective` from a parameter dictionary.

        Required keys are:
        - ``width``
        - ``height``

        Optional keys are:
        - ``position`` (3D)
        - ``orientation_matrix`` (3x3)
        - ``crs`` (WKT)
        - ``camera`` (camera ``param_dict``)

        :param param_dict: Dictionary with image parameters.
        :type param_dict: dict
        :param mapper: Mapping instance, defaults to ``None``.
        :type mapper: MappingBase | None
        :return: Image model instance.
        :rtype: ImagePerspective
        :raises KeyError: If required keys are missing.
        :raises ValueError: If values are invalid.
        :raises TypeError: If values have incompatible types.
        :raises CRSInputError: If the CRS WKT string is invalid or unsupported.
        """

        width = param_dict["width"]
        height = param_dict["height"]

        if width == 0 or height == 0:
            raise ValueError("Image width and height can not be 0")

        position = None
        if param_dict.get("position") is not None:
            position = np.array(param_dict.get("position"))

        orientation = None
        if param_dict.get("orientation_matrix") is not None:
            r_matrix = np.array(param_dict.get("orientation_matrix"))
            orientation = Rotation(rotation_matrix=r_matrix)

        crs = None
        if param_dict.get("crs") is not None:
            try:
                crs = CRS(param_dict["crs"])
            except pyproj.exceptions.CRSError as err:
                raise CRSInputError("No valid CRS wkt string supported") from err

        camera = None
        if param_dict.get("camera") is not None:
            camera = get_camera_from_dict(param_dict["camera"])

        image = cls(
            width=width,
            height=height,
            camera=camera,
            position=position,
            orientation=orientation,
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
            "type": ImageType.Perspective.fullname,
            "width": self.width,
            "height": self.height,
            "position": self._position.tolist() if self._position is not None else None,
            "orientation_matrix": self._orientation.matrix.tolist() if self._orientation is not None else None,
            "camera": self.camera.param_dict if self.camera is not None else None,
            "crs": self.crs_wkt,
        }

        return param_dict

    @property
    def is_geo_referenced(self) -> bool:
        """Return whether the image is geo-referenced.

        :return: ``True`` if position, orientation and camera are set, otherwise ``False``.
        :rtype: bool
        """
        if self._position is not None and self._orientation is not None and self._camera is not None:
            return True
        return False

    @property
    def camera(self):
        """Return the camera model used by this image.

        :return: Camera model instance or ``None``.
        :rtype: CameraBasePerspective | None
        """
        return self._camera

    @property
    def center(self) -> tuple[float, float]:
        """Return the image center used for mapping/projection.

        For perspective images this is the principal point as defined by the camera model.

        :return: Center point (x, y) in image pixel coordinates.
        :rtype: tuple[float, float]
        :raises NotGeoreferencedError: If no camera model is specified.
        """

        if self._camera is None:
            raise NotGeoreferencedError("camera is not specified")
        center = self._camera.principal_point(self.shape)
        return float(center[0]), float(center[1])

    def image_points_inside(self, point_image_coordinates: ArrayNx2) -> MaskN_:
        """Return whether undistorted image points are valid for this camera model.

        Within the camera class the distortion border is generated during initialization and
        is used to determine where distortion/undistortion is valid.

        :param point_image_coordinates: Image pixel coordinates (image CRS).
        :type point_image_coordinates: ArrayNx2
        :return: Boolean mask of shape ``(N,)``.
        :rtype: ``MaskN_``
        :raises NotGeoreferencedError: If no camera model is specified.
        """

        if self._camera is None:
            raise NotGeoreferencedError("camera is not specified")
        return self._camera.undistorted_image_points_inside(point_image_coordinates, self.shape)

        # old version using discarded parameter outer dist
        # min_border = np.logical_and(point_image_coordinates[:, 0] >= -outer_dist,
        #                            point_image_coordinates[:, 1] >= -outer_dist)
        # max_border = np.logical_and(point_image_coordinates[:, 0] < self.width + outer_dist,
        #                            point_image_coordinates[:, 1] < self.height + outer_dist)
        # valid_index = np.where(np.logical_and(min_border, max_border))[0]
        #
        # if not valid_index.size:
        #    return None
        # return valid_index

    def position_to_crs(self, crs_t: CRS) -> Vector3D | None:
        """Return the camera position in the target CRS.

        :param crs_t: Target CRS.
        :type crs_t: CRS
        :return: Position in ``crs_t`` or ``None`` if unavailable.
        :rtype: Vector3D | None
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """
        if self._position is not None and self._crs is not None:
            coo_trafo = CoordinateTransformer.from_crs(crs_s=self._crs, crs_t=crs_t)

            if coo_trafo is not None:
                coordinates_crs_t = coo_trafo.transform(self._position)
            else:
                coordinates_crs_t = self._position * 1.0

            return coordinates_crs_t[0, :]
        return None

    def project(
        self,
        coordinates: ArrayNx3,
        crs_s: CRS | None = None,
        transformer: CoordinateTransformer | None = None,
        to_distorted: bool = True,
    ) -> ProjectionResult:
        """Project 3D coordinates into image pixel coordinates.

        The projection is first computed in undistorted image coordinates. If ``to_distorted`` is ``True``,
        only pixels within the camera model's valid undistortion border are distorted; invalid pixels remain
        undistorted and are marked in the returned mask.

        :param coordinates: 3D coordinates to project.
        :type coordinates: ArrayNx3
        :param crs_s: CRS of the input coordinates, defaults to ``None``.
        :type crs_s: CRS | None
        :param transformer: Optional coordinate transformer, defaults to ``None``.
        :type transformer: CoordinateTransformer | None
        :param to_distorted: Whether to return distorted image coordinates, defaults to ``True``.
        :type to_distorted: bool
        :return: Projection result.
        :rtype: ProjectionResult
        :raises NotGeoreferencedError: If the image is not geo-referenced.
        :raises CRSInputError: If both ``crs_s`` and ``transformer`` are provided.
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """

        points_world_crs = to_array_nx3(coordinates)

        if not self.is_geo_referenced:
            raise NotGeoreferencedError()

        # Transformer
        if crs_s is not None and transformer is not None:
            raise CRSInputError("Either crs or transformation can be used or both None")
        coo_trafo = (
            transformer if transformer is not None else CoordinateTransformer.from_crs(crs_s=crs_s, crs_t=self._crs)
        )

        if coo_trafo is not None:
            coordinates_image_crs = coo_trafo.transform(points_world_crs)
        else:
            coordinates_image_crs = points_world_crs * 1.0

        assert self._orientation is not None
        assert self._position is not None
        # Bring 3d points into Camera CRS by using rotation matrix and projection center
        pts_camera_crs = np.matmul(self._orientation.matrix.T, (coordinates_image_crs - self._position).T).T

        assert self._camera is not None
        pts_image_crs = self._camera.pts_camara_crs_to_image_pixel(pts_camera_crs, self.shape, to_distorted=False)

        # Here we check if the undistorted image point is inside the border for which distorted points are inside
        # This is important as non-linear function of distortion,
        # would map extreme far points back to the image due to the higher order polynomials
        #
        valid_pixel = self._camera.undistorted_image_points_inside(pts_image_crs, image_size=self.shape)

        # Only the valid pixel which are inside the distortion border are distorted
        pts_image_crs[valid_pixel, :] = self._camera.pts_camara_crs_to_image_pixel(
            pts_camera_crs[valid_pixel, :], self.shape, to_distorted=to_distorted
        )

        issue = set()
        if not np.all(valid_pixel):
            issue = {Issue.INVALID_PROJECTIIONS}
            if not np.any(valid_pixel):
                return ResultFailure(ok=False, error="None of the projections is valid", issues=issue)

        # return pts_image_crs, valid_pixel
        return ProjectionResultSuccess(ok=True, pixels=pts_image_crs, mask=valid_pixel, issues=issue)

    def pixel_to_ray_vector(self, pixel_pos: ArrayNx2, is_undistorted: bool = False) -> ArrayNx3:
        """Calculate world ray direction vectors for image pixels.

        :param pixel_pos: Image pixel coordinates (pixel CRS).
        :type pixel_pos: ArrayNx2
        :param is_undistorted: Whether input pixels are already undistorted, defaults to ``False``.
        :type is_undistorted: bool
        :return: Unit direction vectors in world CRS.
        :rtype: ArrayNx3
        :raises NotGeoreferencedError: If the image is not geo-referenced.
        """

        _pixel_pos = to_array_nx2(pixel_pos)

        if not self.is_geo_referenced:
            raise NotGeoreferencedError("Image is not georeferenced")

        assert self._camera is not None
        pts_camera_crs = self._camera.pixel_image_to_camera_crs(_pixel_pos, self.shape, is_undistorted)

        assert self._orientation is not None
        line_vec = np.matmul(self._orientation.matrix, pts_camera_crs.T).T
        line_vec = np.divide(line_vec.T, np.linalg.norm(line_vec, axis=1)).T

        return line_vec

    def map_center_point(
        self, mapper: MappingBase | None = None, transformer: CoordinateTransformer | None = None
    ) -> MappingResult:
        """Map the image center pixel (principal point) to 3D coordinates.

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
        :raises MappingBackendError: If the mapping backend fails unexpectedly.
        :raises WeitsichtError: Base class for all weitsicht exceptions. Catch this to handle any weitsicht error;
            catch specific subclasses first if you need to distinguish causes.
        """

        if not self.is_geo_referenced:
            raise NotGeoreferencedError()

        if self.mapper is None and mapper is None:
            raise MapperMissingError("No mapper provided for map_center_point")

        mapper_to_use = self.mapper
        if mapper is not None:
            mapper_to_use = mapper

        center_pixel = np.array([self.center], dtype=float)
        ray_vector = self.pixel_to_ray_vector(center_pixel)

        # checked pos and camera via is_geo_referenced
        assert mapper_to_use is not None
        assert self._position is not None
        assert self._camera is not None

        try:
            result_mapper = mapper_to_use.map_coordinates_from_rays(
                ray_vector,
                np.array([self._position]),
                None if transformer is not None else self._crs,
                transformer=transformer,
            )
        except WeitsichtError:
            raise
        except Exception as err:
            raise MappingBackendError(f"{type(mapper_to_use).__name__}.map_coordinates_from_rays failed") from err

        if result_mapper.ok is False:
            return ResultFailure(ok=False, error=result_mapper.error, issues=result_mapper.issues)

        coordinates = result_mapper.coordinates
        gsd, gsd_per_point = self._estimate_gsd_for_mapped_points(
            pixels=center_pixel,
            ray_vectors=ray_vector,
            coordinates=coordinates,
            normals=result_mapper.normals,
            mask=result_mapper.mask,
            is_undistorted=False,
        )
        return MappingResultSuccess(
            ok=True,
            coordinates=coordinates,
            mask=result_mapper.mask,
            normals=result_mapper.normals,
            crs=self._crs,
            issues=result_mapper.issues,
            gsd=gsd,
            gsd_per_point=gsd_per_point,
        )

    def map_footprint(
        self,
        points_per_edge: int = 0,
        mapper: MappingBase | None = None,
        transformer: CoordinateTransformer | None = None,
    ) -> MappingResult:
        """Map the image footprint polygon to 3D coordinates.

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
        :raises MappingBackendError: If the mapping backend fails unexpectedly.
        :raises WeitsichtError: Base class for all weitsicht exceptions. Catch this to handle any weitsicht error;
            catch specific subclasses first if you need to distinguish causes.
        """

        if not self.is_geo_referenced:
            raise NotGeoreferencedError()

        if self.mapper is None and mapper is None:
            raise MapperMissingError("No mapper provided for map_footprint")

        mapper_to_use = self.mapper
        if mapper is not None:
            mapper_to_use = mapper

        assert mapper_to_use is not None
        assert self._position is not None
        assert self._camera is not None

        if points_per_edge < 1:
            footprint_points_2d = np.array([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]])
        else:
            footprint_points_2d = [[x, 0] for x in np.linspace(0, self.width, 2 + points_per_edge)]
            footprint_points_2d += [[self.width, x] for x in np.linspace(0, self.height, 2 + points_per_edge)][1:]
            footprint_points_2d += [[x, self.height] for x in np.linspace(self.width, 0, 2 + points_per_edge)][1:]
            footprint_points_2d += [[0, x] for x in np.linspace(self.height, 0, 2 + points_per_edge)][1:-1]

        footprint_points_2d = np.array(footprint_points_2d)
        ray_vector = self.pixel_to_ray_vector(footprint_points_2d)

        ray_pos = np.ones(ray_vector.shape) * self._position

        try:
            result_mapper = mapper_to_use.map_coordinates_from_rays(
                ray_vector, ray_pos, None if transformer is not None else self._crs, transformer=transformer
            )
        except WeitsichtError:
            raise
        except Exception as err:
            raise MappingBackendError(f"{type(mapper_to_use).__name__}.map_coordinates_from_rays failed") from err

        if result_mapper.ok is False:
            return ResultFailure(ok=False, error=result_mapper.error, issues=result_mapper.issues)

        if not np.all(result_mapper.mask):
            return ResultFailure(
                ok=False,
                error="Some footprint points failed",
                issues=result_mapper.issues,
            )

        footprint_points_3d = result_mapper.coordinates

        footprint_geom = geometry.Polygon(footprint_points_3d)
        area = float(np.round(footprint_geom.area))
        # gsd = float(np.round(np.sqrt(area / (self.width * self.height)), 4))

        gsd, gsd_per_point = self._estimate_gsd_for_mapped_points(
            pixels=footprint_points_2d,
            ray_vectors=ray_vector,
            coordinates=footprint_points_3d,
            normals=result_mapper.normals,
            mask=result_mapper.mask,
            is_undistorted=False,
        )

        return MappingResultSuccess(
            ok=True,
            coordinates=footprint_points_3d,
            mask=result_mapper.mask,
            normals=result_mapper.normals,
            crs=self._crs,
            gsd=gsd,
            gsd_per_point=gsd_per_point,
            area=area,
            issues=result_mapper.issues,
        )

    def map_points(
        self,
        points_image: ArrayNx2 | ArrayNx3 | list[list[float]] | list[list[int]] | list[float] | list[int],
        mapper: MappingBase | None = None,
        transformer: CoordinateTransformer | None = None,
        is_undistorted: bool = False,
    ) -> MappingResult:
        """Map image pixel coordinates to 3D coordinates via a mapper.

        It is also possible to pass undistorted pixels. This can be useful if you already computed
        undistorted pixel coordinates from the original distorted image content.

        :param points_image: Pixel coordinates (distorted or undistorted).
        :type points_image: ArrayNx2 | ArrayNx3 | list[list[float]] | list[list[int]] | list[float] | list[int]
        :param mapper: Mapper to use, defaults to ``None`` (uses :attr:`~weitsicht.image.base_class.ImageBase.mapper`).
        :type mapper: MappingBase | None
        :param transformer: Optional coordinate transformer, defaults to ``None``.
        :type transformer: CoordinateTransformer | None
        :param is_undistorted: Whether input pixels are already undistorted, defaults to ``False``.
        :type is_undistorted: bool
        :return: Mapping result.
        :rtype: MappingResult
        :raises ValueError: If ``points_image`` cannot be parsed as an array of 2D points.
        :raises NotGeoreferencedError: If the image is not geo-referenced.
        :raises MapperMissingError: If no mapper is available.
        :raises CRSInputError: If the mapper rejects CRS/transformer input.
        :raises CoordinateTransformationError: If coordinate transformation fails.
        :raises MappingBackendError: If the mapping backend fails unexpectedly.
        :raises WeitsichtError: Base class for all weitsicht exceptions. Catch this to handle any weitsicht error;
            catch specific subclasses first if you need to distinguish causes.
        """
        # TODO what should we do if specified points_image are outside the image dimensions
        # but the mapping still works? Should be masked them as invalid?
        # different to the orthoimage here far outside pixel coordinates could be distorted back
        # to valid areas.

        if not self.is_geo_referenced:
            raise NotGeoreferencedError()

        if self.mapper is None and mapper is None:
            raise MapperMissingError("No mapper provided for map_points")

        mapper_to_use = self.mapper
        if mapper is not None:
            mapper_to_use = mapper

        try:
            _points_image = to_array_nx2(points_image)
        except (IndexError, TypeError, ValueError) as err:
            raise ValueError("points_image must be array-like with shape (N, 2) or (N, 3)") from err
        # build ray vectors of image points
        ray_vector = self.pixel_to_ray_vector(_points_image, is_undistorted=is_undistorted)
        # build ray start points for all rays
        assert self._position is not None
        ray_pos = np.ones(ray_vector.shape) * self._position

        assert mapper_to_use is not None
        # Map points using either the images specified mapper or the user mapper provided

        try:
            mapping_result: MappingResult = mapper_to_use.map_coordinates_from_rays(
                ray_vector, ray_pos, None if transformer is not None else self._crs, transformer=transformer
            )
        except WeitsichtError:
            raise
        except Exception as err:
            raise MappingBackendError(f"{type(mapper_to_use).__name__}.map_coordinates_from_rays failed") from err

        if mapping_result.ok is False:
            return ResultFailure(ok=False, error=mapping_result.error, issues=mapping_result.issues)

        gsd, gsd_per_point = self._estimate_gsd_for_mapped_points(
            pixels=_points_image,
            ray_vectors=ray_vector,
            coordinates=mapping_result.coordinates,
            normals=mapping_result.normals,
            mask=mapping_result.mask,
            is_undistorted=is_undistorted,
        )
        return MappingResultSuccess(
            ok=True,
            coordinates=mapping_result.coordinates,
            mask=mapping_result.mask,
            normals=mapping_result.normals,
            gsd=gsd,
            gsd_per_point=gsd_per_point,
            issues=mapping_result.issues,
        )

    def _estimate_gsd_for_mapped_points(
        self,
        pixels: ArrayNx2,
        ray_vectors: ArrayNx3,
        coordinates: ArrayNx3,
        normals: ArrayNx3,
        mask: MaskN_,
        is_undistorted: bool,
        *,
        pixel_step: float = 1.0,
    ) -> tuple[float | None, ArrayN_]:
        """Estimate GSD for mapped points.

        Computes per-point GSD from the distance to each mapped 3D point and the angular
        separation of neighbouring pixel rays (no additional mapper calls).

        Surface normals (in the same CRS as the mapped coordinates) are used to correct
        for oblique viewing angles. If normals are valid, neighbouring pixel rays are
        intersected with the local tangent plane at the mapped point. As a fallback,
        the range-sphere chord estimate is scaled by ``1 / cos(incidence)`` where
        ``incidence`` is the angle between the viewing direction and the surface normal.
        """

        if not self.is_geo_referenced:
            raise NotGeoreferencedError("Image is not georeferenced")

        assert self._position is not None
        assert self._camera is not None

        if pixel_step <= 0:
            raise ValueError("pixel_step must be > 0")

        _pixels = to_array_nx2(pixels).astype(np.float64, copy=False)
        _ray_vectors = to_array_nx3(ray_vectors).astype(np.float64, copy=False)
        _coordinates = to_array_nx3(coordinates).astype(np.float64, copy=False)
        _normals = to_array_nx3(normals).astype(np.float64, copy=False)
        n_points = _pixels.shape[0]

        if _ray_vectors.shape[0] != n_points:
            raise ValueError("pixels and ray_vectors must have the same length")
        if _coordinates.shape[0] != n_points:
            raise ValueError("pixels and coordinates must have the same length")
        if _normals.shape[0] != n_points:
            raise ValueError("pixels and normals must have the same length")

        gsd_per_point = np.full((n_points,), np.nan, dtype=np.float64)

        # Fallback: approximate per-point GSD via range/focal length (still varies with distance).
        # If surface normals are available, scale by 1 / cos(incidence) to approximate the
        # footprint on the local tangent plane at the intersection point.
        view_vec = self._position - _coordinates
        dist = np.linalg.norm(view_vec, axis=1)

        # Default to no correction (cos=1) for missing/invalid normals.
        cos_incidence = np.ones((n_points,), dtype=np.float64)

        normal_len = np.linalg.norm(_normals, axis=1)
        valid_normals = np.isfinite(_normals).all(axis=1) & (normal_len > 0.0)
        valid_view = np.isfinite(view_vec).all(axis=1) & np.isfinite(dist) & (dist > 0.0)

        normals_unit = np.full_like(_normals, np.nan)
        if np.any(valid_normals):
            normals_unit[valid_normals, :] = _normals[valid_normals, :] / normal_len[valid_normals, None]

        view_unit = np.full_like(view_vec, np.nan)
        if np.any(valid_view):
            view_unit[valid_view, :] = view_vec[valid_view, :] / dist[valid_view, None]

        valid_inc = valid_normals & valid_view & mask
        min_cos = 1e-6
        if np.any(valid_inc):
            cos_raw = np.abs(np.sum(normals_unit[valid_inc, :] * view_unit[valid_inc, :], axis=1))
            cos_incidence[valid_inc] = np.clip(cos_raw, min_cos, 1.0)

        gsd_approx = (dist / self._camera.focal_length_for_gsd_in_pixel) / cos_incidence
        gsd_per_point[mask] = gsd_approx[mask]

        # Neighbour rays: choose direction so that we stay inside image bounds for border points.
        step_x = np.where(_pixels[:, 0] >= self.width - pixel_step, -pixel_step, pixel_step)
        step_y = np.where(_pixels[:, 1] >= self.height - pixel_step, -pixel_step, pixel_step)

        pixels_dx = _pixels.copy()
        pixels_dy = _pixels.copy()
        pixels_dx[:, 0] += step_x
        pixels_dy[:, 1] += step_y

        try:
            ray_vec_neighbours = self.pixel_to_ray_vector(
                np.vstack((pixels_dx, pixels_dy)), is_undistorted=is_undistorted
            )
        except Exception:
            # Best effort only: keep approximate GSDs.
            valid_mean = mask & np.isfinite(gsd_per_point)
            gsd_mean = float(np.mean(gsd_per_point[valid_mean])) if np.any(valid_mean) else None
            return gsd_mean, gsd_per_point

        ray_vec_dx = ray_vec_neighbours[:n_points, :]
        ray_vec_dy = ray_vec_neighbours[n_points:, :]

        # Advanced GSD: chord length between rays evaluated at the point distance.
        valid_adv = mask & np.isfinite(dist) & (dist > 0)
        if np.any(valid_adv):
            # Fallback: use chord length on the range sphere, corrected by incidence angle.
            gsd_x_chord = dist * np.linalg.norm(ray_vec_dx - _ray_vectors, axis=1) / np.abs(step_x)
            gsd_y_chord = dist * np.linalg.norm(ray_vec_dy - _ray_vectors, axis=1) / np.abs(step_y)
            gsd_adv = ((gsd_x_chord + gsd_y_chord) / 2.0) / cos_incidence

            # If normals are valid, intersect neighbour rays with the local tangent plane at the
            # mapped point (uses the normal at the intersection).
            valid_plane = valid_adv & valid_normals
            if np.any(valid_plane):
                # Plane: (x - p)·n = 0, with p=intersection point and n=unit normal.
                # Ray: x = c + t*u, with c=camera position and u=ray direction.
                # Solve: t = (p - c)·n / (u·n)
                cam_pos = np.asarray(self._position, dtype=float)
                cam_points = np.broadcast_to(cam_pos, (n_points, 3))

                p_dx, valid_dx_raw = intersection_plane_mat_operation(
                    line_vec=ray_vec_dx,
                    line_point=cam_points,
                    plane_point=_coordinates,
                    plane_normal=normals_unit,
                )
                p_dy, valid_dy_raw = intersection_plane_mat_operation(
                    line_vec=ray_vec_dy,
                    line_point=cam_points,
                    plane_point=_coordinates,
                    plane_normal=normals_unit,
                )
                valid_dx = valid_dx_raw & valid_plane
                valid_dy = valid_dy_raw & valid_plane

                gsd_x_plane = np.full((n_points,), np.nan, dtype=np.float64)
                gsd_y_plane = np.full((n_points,), np.nan, dtype=np.float64)

                if np.any(valid_dx):
                    gsd_x_plane[valid_dx] = np.linalg.norm(
                        p_dx[valid_dx, :] - _coordinates[valid_dx, :], axis=1
                    ) / np.abs(step_x[valid_dx])

                if np.any(valid_dy):
                    gsd_y_plane[valid_dy] = np.linalg.norm(
                        p_dy[valid_dy, :] - _coordinates[valid_dy, :], axis=1
                    ) / np.abs(step_y[valid_dy])

                gsd_plane = np.full((n_points,), np.nan, dtype=np.float64)
                both = np.isfinite(gsd_x_plane) & np.isfinite(gsd_y_plane)
                gsd_plane[both] = (gsd_x_plane[both] + gsd_y_plane[both]) / 2.0

                only_x = np.isfinite(gsd_x_plane) & ~np.isfinite(gsd_y_plane)
                gsd_plane[only_x] = gsd_x_plane[only_x]

                only_y = np.isfinite(gsd_y_plane) & ~np.isfinite(gsd_x_plane)
                gsd_plane[only_y] = gsd_y_plane[only_y]

                gsd_adv = np.where(np.isfinite(gsd_plane), gsd_plane, gsd_adv)

            valid_adv &= np.isfinite(gsd_adv)
            gsd_per_point[valid_adv] = gsd_adv[valid_adv]

        valid_mean = mask & np.isfinite(gsd_per_point)
        gsd_mean = float(np.mean(gsd_per_point[valid_mean])) if np.any(valid_mean) else None
        return gsd_mean, gsd_per_point
