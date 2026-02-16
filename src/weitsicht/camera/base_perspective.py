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

"""Base API for camera perspective models.

Defines :class:`~weitsicht.camera.base_perspective.CameraBasePerspective`, which
converts between image pixel coordinates and camera coordinate system vectors,
including distortion/undistortion handling.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import cast

import numpy as np
from shapely import geometry

from weitsicht.camera.camera_types import CameraType
from weitsicht.utils import (
    ArrayNx2,
    ArrayNx3,
    MaskN_,
    Vector2D,
    to_array_nx2,
    to_array_nx3,
)

__all__ = ["CameraBasePerspective"]


class PostMetaCaller(type):
    def __call__(cls, *args, **kwargs):
        # here is "before __new__ is called"
        instance = type.__call__(cls, *args, **kwargs)
        instance.__post_init__()

        return instance


class CameraBasePerspective(metaclass=PostMetaCaller):
    """Base class for perspective camera models.

    This class converts between image pixel coordinates ("image CRS") and rays in
    camera coordinates ("camera CRS"). It does not store geo-referencing; world
    coordinate transforms are handled by the image layer.

    Coordinate system conventions:
    - Image pixel CRS: x points right, y points down (origin at top-left pixel edge).
    - Camera CRS: x points right, y points up, z points backwards.

    Implementations typically apply origin shifts and scale pixel coordinates to
    a calibration size so that distortion parameters remain valid for resampled
    images.

    Subclasses must implement:
    - `from_dict`
    - `type`
    - `param_dict`
    - `focal_length_for_gsd_in_pixel`
    - `_origin`
    - `_principal_point`
    - `_to_distorted_points`
    - `_to_undistorted_points`
    - `_image_crs_to_vector_in_camera_crs`
    - `_camara_crs_to_image_crs`
    """

    def __init__(self, width: int, height: int, pts_distortion: int = 5):
        """Initialize a camera perspective model base.

        :param width: Calibration image width in pixels.
        :type width: int
        :param height: Calibration image height in pixels.
        :type height: int
        :param pts_distortion: Number of points used to generate the distortion border, defaults to ``5``.
        :type pts_distortion: int
        """

        self.calibration_width = width
        self.calibration_height = height

        # initial border
        self.distortion_border = geometry.Polygon(
            [
                [0, 0],
                [self.calibration_width, 0],
                [self.calibration_width, self.calibration_height],
                [0, self.calibration_height],
            ]
        )
        self.pts_distortion = pts_distortion

    def __post_init__(self):
        """Finalize initialization.

        Called automatically after initialization (including subclasses) to
        compute the polygon that bounds valid undistorted points.
        """
        distortion_border = self._generate_distortion_border(self.pts_distortion)
        self.distortion_border = geometry.Polygon(distortion_border)

    @classmethod
    @abstractmethod
    def from_dict(cls, param_dict: dict) -> CameraBasePerspective:
        """Create a camera model from a parameter dictionary.

        :param param_dict: Dictionary as produced by :attr:`param_dict`.
        :type param_dict: dict
        :return: Camera model instance.
        :rtype: CameraBasePerspective
        """
        # Has to be implemented for each camera Class
        pass

    @property
    @abstractmethod
    def param_dict(self) -> dict:
        """Return camera parameters as a dictionary.

        The returned dictionary must be compatible with :meth:`from_dict`.

        :return: Camera parameters.
        :rtype: dict
        """
        # Has to be implemented for each camera Class
        pass

    @property
    @abstractmethod
    def type(self) -> CameraType:
        """Return the camera model type.

        :return: Camera model type.
        :rtype: CameraType
        """
        # Has to be implemented for each camera Class
        ...

    @property
    @abstractmethod
    def focal_length_for_gsd_in_pixel(self) -> float:
        """Returns the focal length in pixel to be used for approximate gsd calculation

        :return: Focal length in pixel.
        :rtype: float
        """
        # Has to be implemented for each camera Class
        pass

    @property
    @abstractmethod
    def _origin(self) -> Vector2D:
        """One has to implement the pixel origin of the camera model as offset to the used
        pixel system. Our system is using the upper left corner as 0,0
        For example OpenCVs pixel coordinate system is defined with left upper CENTER is (0,0) therefore
        we need to apply an offset and return np.array([0.5,0.5])

        :return: Origin offset in pixels.
        :rtype: Vector2D
        """
        pass

    @property
    @abstractmethod
    def _principal_point(self) -> Vector2D:
        """principal point in camera size

        :return: Principal point in camera calibration size.
        :rtype: Vector2D
        """
        # Has to be implemented for each camera Class
        pass

    def _generate_distortion_border(self, points_between: int = 5) -> ArrayNx2:
        """Returns the border of the undistorted image, which is used as border for points which
        are valid to calculate the distorted ones (The distorted border is the rectangular image itself)

        :param points_between: Number of points between corner points, defaults to ``5``.
        :type points_between: int
        :return: ArrayNx2 with the undistorted pixel of the border points
        :rtype: ArrayNx2
        """

        img_points = np.vstack(
            (
                np.linspace(
                    [0, 0],
                    [self.calibration_width, 0],
                    points_between + 1,
                    axis=0,
                    endpoint=False,
                ),
                np.linspace(
                    [self.calibration_width, 0],
                    [self.calibration_width, self.calibration_height],
                    points_between + 1,
                    axis=0,
                    endpoint=False,
                ),
                np.linspace(
                    [self.calibration_width, self.calibration_height],
                    [0, self.calibration_height],
                    points_between + 1,
                    axis=0,
                    endpoint=False,
                ),
                np.linspace(
                    [0, self.calibration_height],
                    [0, 0],
                    points_between + 1,
                    axis=0,
                    endpoint=False,
                ),
            )
        )

        img_points = cast(ArrayNx2, img_points)
        distortion_border = self.distorted_to_undistorted(img_points, (self.calibration_width, self.calibration_height))

        return distortion_border

    @abstractmethod
    def _to_distorted_points(self, pts_undistorted_pixel: ArrayNx2) -> ArrayNx2:
        """function for calculating distorted points from undistorted ones in original calibration image size

        :param pts_undistorted_pixel: Array of (nx2) with the image coordinates
        :type pts_undistorted_pixel: ArrayNx2
        :return: Array with distorted points
        :rtype: ArrayNx2
        """
        # Has to be implemented for each camera Class
        pass

    @abstractmethod
    def _to_undistorted_points(self, pts_distorted_pixel: ArrayNx2) -> ArrayNx2:
        """function for calculating undistorted points from distorted ones

        :param pts_distorted_pixel: Array of (nx2) with the image coordinates
        :type pts_distorted_pixel: ArrayNx2
        :return: ArrayNx2 with undistorted points
        :rtype: ArrayNx2
        """
        # Has to be implemented for each camera Class
        pass

    @abstractmethod
    def _image_crs_to_vector_in_camera_crs(self, pts_image: ArrayNx2) -> ArrayNx3:
        """function for calculating vector in camera crs of undistorted image points

        :param pts_image: Array of (nx2) with the image coordinates
        :type pts_image: ArrayNx2
        :return: Array of (Nx3) with vectors of the line of sight of image positions
        :rtype: ArrayNx3
        """
        # Has to be implemented for each camera Class
        pass

    @abstractmethod
    def _camara_crs_to_image_crs(self, pts_camera_crs: ArrayNx3) -> ArrayNx2:
        """function for calculating undistorted points from points in camera crs

        :param pts_camera_crs: Array of (nx3) with the camera CRS coordinates
        :type pts_camera_crs: ArrayNx3
        :return: Array of size (N,2) with undistorted points
        :rtype: ArrayNx2
        """
        # Has to be implemented for each camera Class
        pass

    @property
    def origin(self) -> Vector2D:
        return self._origin

    def principal_point(self, image_size: tuple[int, int]) -> Vector2D:
        """Return the principal point in the given image size.

        :param image_size: Size of the current image (width, height).
        :type image_size: tuple[int, int]
        :return: Principal point in image pixel CRS.
        :rtype: Vector2D
        """
        return self._pixel_calibration_to_image_size(self._principal_point, image_size)[0]

    def _pixel_image_to_calibration_size(self, pixel_from_image: ArrayNx2, image_size: tuple[int, int]) -> ArrayNx2:
        """Calculate scaled image coordinates to calibration size

        :param pixel_from_image: Array of (nx2) with the image coordinates
        :type pixel_from_image: ArrayNx2
        :param image_size: Size of the current image (width, height).
        :type image_size: tuple[int, int]
        :return: Array with the image positions scaled to calibration image size
        :rtype: ArrayNx2
        """
        _image_size = np.array(image_size)
        _pixel_from_image = to_array_nx2(pixel_from_image)

        px_normalized = _pixel_from_image / _image_size
        px_calibration_size = px_normalized * np.array([self.calibration_width, self.calibration_height])
        px_calibration_size -= self._origin
        return px_calibration_size

    def _pixel_calibration_to_image_size(self, pixel_from_calib_size: ArrayNx2, image_size: tuple[int, int]):
        """Calculate scaled image coordinates to image file size

        :param pixel_from_calib_size: Array of (nx2) with the image coordinates
        :type pixel_from_calib_size: ArrayNx2
        :param image_size: Size of the current image (width, height).
        :type image_size: tuple[int, int]
        :return: Array with the image positions scaled to file image size
        :rtype: ArrayNx2
        """
        _image_size = np.asarray(image_size)

        pixel_from_calib_size = to_array_nx2(pixel_from_calib_size)

        img_calib_size = np.array([self.calibration_width, self.calibration_height])
        px_normalized = pixel_from_calib_size / img_calib_size
        px_image_size = px_normalized * _image_size

        # Correct px image size for origin shift from the camera model
        px_image_size += self.origin
        return px_image_size

    def undistorted_image_points_inside(self, points_image_crs: ArrayNx2, image_size: tuple[int, int]) -> MaskN_:
        """Return a mask indicating whether undistorted image points are valid.

        This takes the distortion border of the camera model into account.

        :param points_image_crs: Array of (nx2) with image pixel coordinates.
        :type points_image_crs: ArrayNx2
        :param image_size: Size of the current image (width, height).
        :type image_size: tuple[int, int]
        :return: Boolean mask of shape (N,).
        :rtype: ``MaskN_``
        """

        pixel_calibration_size = self._pixel_image_to_calibration_size(points_image_crs, image_size=image_size)

        valid_mask = np.full((pixel_calibration_size.shape[0]), False)
        for idx, pt in enumerate(pixel_calibration_size):
            valid_mask[idx] = self.distortion_border.contains(geometry.Point(pt + self.origin))

        return valid_mask

    def pixel_image_to_camera_crs(
        self,
        points_image_crs: ArrayNx2,
        image_size: tuple[int, int],
        is_undistorted: bool = False,
    ) -> ArrayNx3:
        """Calculating vectors in camera crs of image points.
        Image points are internally scaled to calibration image size for validity of distortion parameters

        :param points_image_crs: Array of (nx2) with the image coordinates
        :type points_image_crs: ArrayNx2
        :param image_size: Size of the current image (width, height).
        :type image_size: tuple[int, int]
        :param is_undistorted: Whether input image coordinates are already undistorted, defaults to ``False``.
        :type is_undistorted: bool
        :return: Array of size (N,3) with vectors of the line of sight of image positions
        :rtype: ArrayNx3
        """
        points_image_crs = to_array_nx2(points_image_crs)

        points_image_scaled_to_calibration = self._pixel_image_to_calibration_size(points_image_crs, image_size)

        # Points will be undistorted.
        # points_image are in our pixel definition
        if not is_undistorted:
            points_image_scaled_to_calibration = self._to_undistorted_points(points_image_scaled_to_calibration)

        # Points will be scaled to calibration size as the camera class has its own pixel system definition
        pts_camera_crs = self._image_crs_to_vector_in_camera_crs(points_image_scaled_to_calibration)
        return pts_camera_crs

    def pts_camara_crs_to_image_pixel(
        self, points_camera_crs: ArrayNx3, image_size: tuple[int, int], to_distorted: bool = True
    ) -> ArrayNx2:
        """Calculating image pixel coordinates of points in camera crs.
        Image points are internally scaled to calibration image size for validity of distortion parameters

        :param points_camera_crs: Array of (nx3) with the camera crs coordinates
        :type points_camera_crs: ArrayNx3
        :param image_size: Size of the current image (width, height).
        :type image_size: tuple[int, int]
        :param to_distorted: Whether to return distorted image coordinates, defaults to ``True``.
        :type to_distorted: bool
        :return: Array of size (N,2) with pixels in image coordinates. Undistorted or distorted according to_distorted
                 and the camera model used.
        :rtype: ArrayNx2
        """
        points_camera_crs = to_array_nx3(points_camera_crs)

        points_image_in_camera_calib_size = self._camara_crs_to_image_crs(points_camera_crs)

        if to_distorted:
            # Here we are still in the system defined by the child camera
            points_image_in_camera_calib_size = self._to_distorted_points(points_image_in_camera_calib_size)

        points_image_scaled_to_image = self._pixel_calibration_to_image_size(
            points_image_in_camera_calib_size, image_size
        )

        return points_image_scaled_to_image

    def distorted_to_undistorted(self, points_distorted: ArrayNx2, image_size: tuple[int, int]) -> ArrayNx2:
        """Calculating undistorted image coordinates from distorted ones
        Image points are internally scaled to calibration image size for validity of distortion parameters

        :param points_distorted: Pixel coordinates in pixel CRS
        :type points_distorted: ArrayNx2
        :param image_size: Size of the image pts are from (width, height).
        :type image_size: tuple[int, int]
        :return: Array of size (Nx2) with undistorted image coordinates
        :rtype: ArrayNx2
        """
        scaled_pixel = self._pixel_image_to_calibration_size(points_distorted, image_size)
        undistorted_scaled = self._to_undistorted_points(scaled_pixel)
        undistorted_back_scaled_pixel = self._pixel_calibration_to_image_size(undistorted_scaled, image_size)
        return undistorted_back_scaled_pixel

    def undistorted_to_distorted(self, pts_undistorted: ArrayNx2, image_size: tuple[int, int]) -> ArrayNx2:
        """Calculating distorted image coordinates from undistorted ones
        Image points are internally scaled to calibration image size for validity of distortion parameters

        This function does not prove if the undistorted are inside the border of valid pixels

        :param pts_undistorted: Pixel coordinates in pixel CRS
        :type pts_undistorted: ArrayNx2
        :param image_size: Size of the image pts are from (width, height).
        :type image_size: tuple[int, int]
        :return: Array of size (Nx2) with distorted image coordinates
        :rtype: ArrayNx2
        """
        scaled_pixel = self._pixel_image_to_calibration_size(pts_undistorted, image_size)
        distorted_scaled = self._to_distorted_points(scaled_pixel)
        distorted_back_scaled_pixel = self._pixel_calibration_to_image_size(distorted_scaled, image_size)

        return distorted_back_scaled_pixel
