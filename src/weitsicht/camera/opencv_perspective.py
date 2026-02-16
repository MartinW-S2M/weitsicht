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

"""OpenCV-based pinhole camera model implementation."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from weitsicht.camera.base_perspective import CameraBasePerspective
from weitsicht.camera.camera_types import CameraType
from weitsicht.utils import ArrayNx2, ArrayNx3, Vector2D, to_array_nx3

__all__ = ["CameraOpenCVPerspective"]


logger = logging.getLogger(__name__)


class CameraOpenCVPerspective(CameraBasePerspective):
    """OpenCV perspective (pinhole) camera model.

    See also OpenCV calib3d documentation:
    https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
    """

    def __init__(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float = 0.0,
        cy: float = 0.0,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        k4: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0,
    ):
        """Initialize an OpenCV pinhole camera model.

        :param width: Calibration image width in pixels.
        :type width: int
        :param height: Calibration image height in pixels.
        :type height: int
        :param fx: Focal length in x direction in pixels.
        :type fx: float
        :param fy: Focal length in y direction in pixels.
        :type fy: float
        :param cx: Principal point x coordinate in pixels, defaults to ``0.0`` (interpreted as ``width/2 - 0.5``).
        :type cx: float
        :param cy: Principal point y coordinate in pixels, defaults to ``0.0`` (interpreted as ``height/2 - 0.5``).
        :type cy: float
        :param k1: Radial distortion coefficient, defaults to ``0.0``.
        :type k1: float
        :param k2: Radial distortion coefficient, defaults to ``0.0``.
        :type k2: float
        :param k3: Radial distortion coefficient, defaults to ``0.0``.
        :type k3: float
        :param k4: Radial distortion coefficient, defaults to ``0.0``.
        :type k4: float
        :param p1: Tangential distortion coefficient, defaults to ``0.0``.
        :type p1: float
        :param p2: Tangential distortion coefficient, defaults to ``0.0``.
        :type p2: float
        :raises ValueError: If ``width``, ``height``, ``fx`` or ``fy`` are not positive.
        :raises TypeError: If any parameter has an incompatible type.
        """

        if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
            raise ValueError("with, height, fx and fy are not allowed to be <= 0")

        super().__init__(width=width, height=height)

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # OpenCV is defined as the origin is in the center pixel
        # Therefore center point is shifted 0.5 pixel if no principal point is given
        if self.cx == 0.0:
            self.cx = width / 2 - 0.5
        if self.cy == 0.0:
            self.cy = height / 2 - 0.5

        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.p1 = p1
        self.p2 = p2

    @classmethod
    def from_dict(cls, param_dict: dict) -> CameraOpenCVPerspective:
        """Create a camera model from a parameter dictionary.

        Required keys are:
        - ``calib_width``
        - ``calib_height``
        - ``fx``
        - ``fy``

        Optional keys are:
        - ``cx``, ``cy``
        - ``k1``, ``k2``, ``k3``, ``k4``
        - ``p1``, ``p2``

        :param param_dict: Dictionary with camera parameters.
        :type param_dict: dict
        :return: Camera model instance.
        :rtype: CameraOpenCVPerspective
        :raises KeyError: If required keys are missing.
        :raises ValueError: If values are invalid (e.g. non-positive dimensions/focal lengths).
        :raises TypeError: If values have incompatible types.
        """

        try:
            camera = cls(
                width=param_dict["calib_width"],
                height=param_dict["calib_height"],
                fx=param_dict["fx"],
                fy=param_dict["fy"],
                cx=param_dict.get("cx", 0.0),
                cy=param_dict.get("cy", 0.0),
                k1=param_dict.get("k1", 0.0),
                k2=param_dict.get("k2", 0.0),
                k3=param_dict.get("k3", 0.0),
                k4=param_dict.get("k4", 0.0),
                p1=param_dict.get("p1", 0.0),
                p2=param_dict.get("p2", 0.0),
            )

            return camera
        except (KeyError, ValueError) as err:
            raise err

    @property
    def type(self) -> Literal[CameraType.OpenCV]:
        return CameraType.OpenCV

    @property
    def param_dict(self) -> dict:

        param = {
            "type": self.type.fullname,
            "calib_width": self.calibration_width,
            "calib_height": self.calibration_height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "k1": self.k1,
            "k2": self.k2,
            "k3": self.k3,
            "k4": self.k4,
            "p1": self.p1,
            "p2": self.p2,
        }
        return param

    @property
    def focal_length_for_gsd_in_pixel(self) -> float:
        return float(self.fx / 2.0 + self.fy / 2.0)

    @property
    def _origin(self) -> Vector2D:
        """Return the origin offset for OpenCV pixel conventions.

        OpenCV defines the origin at the center of the top-left pixel, whereas
        this project uses the top-left pixel edge as (0, 0).

        :return: Origin offset in pixels.
        :rtype: Vector2D
        """
        return np.array([0.5, 0.5])

    @property
    def _principal_point(self) -> Vector2D:
        return np.array([self.cx, self.cy])

    def _camara_crs_to_image_crs(self, pts_camera_crs: ArrayNx3) -> ArrayNx2:
        """Project camera CRS coordinates to undistorted image pixel coordinates.

        :param pts_camera_crs: Points in camera CRS.
        :type pts_camera_crs: ArrayNx3
        :return: Undistorted image pixel coordinates.
        :rtype: ArrayNx2
        """

        _pts_camera_crs = to_array_nx3(pts_camera_crs)

        # Change to openCV CRS definition
        _pts_camera_crs[:, 1] = -_pts_camera_crs[:, 1]
        _pts_camera_crs[:, 2] = -_pts_camera_crs[:, 2]

        mat_proj = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0.0, 0.0, 1.0]])

        points_image_crs = np.matmul(mat_proj, _pts_camera_crs.T).T
        points_image_crs = (points_image_crs[:, 0:2].T / points_image_crs[:, 2]).T

        return points_image_crs

    def _image_crs_to_vector_in_camera_crs(self, pts_image: ArrayNx2) -> ArrayNx3:
        """Convert undistorted image pixels to unit vectors in camera CRS.

        :param pts_image: Undistorted image pixel coordinates.
        :type pts_image: ArrayNx2
        :return: Unit direction vectors in camera CRS.
        :rtype: ArrayNx3
        """
        # vector through the pixel
        line_vec = -np.ones((pts_image.shape[0], 3))
        line_vec_xy = pts_image - np.array([self.cx, self.cy])
        line_vec_xy = line_vec_xy / np.array([self.fx, -self.fy])
        line_vec[:, 0:2] = line_vec_xy

        line_vec = np.divide(line_vec.T, np.linalg.norm(line_vec, axis=1)).T

        return line_vec

    def _to_distorted_points(self, pts_undistorted_pixel: ArrayNx2) -> ArrayNx2:
        """Calculate distorted points from undistorted.
        Polynomial functions could map extreme outside points inside due too polynom terms.
        Therefore, we should check for validity first the undistorted points if within border

        :param pts_undistorted_pixel: Undistorted image pixel coordinates.
        :type pts_undistorted_pixel: ArrayNx2
        :return: Distorted image pixel coordinates.
        :rtype: ArrayNx2
        """

        fxy = np.array([self.fx, self.fy])
        p1_p2 = np.array([self.p1, self.p2])
        principle_point = np.array([self.cx, self.cy])

        # As this function is starting from the undistorted we first to some steps to get the x',y' used in opencv
        # x' = Xc/Zc
        # y' = Yc/Zc
        # these are the pts_
        pts_ = np.divide(pts_undistorted_pixel[:, 0:2] - principle_point, fxy)
        # r² = x'² + y'²
        r_2 = np.sum(np.power(pts_, 2), axis=1)
        radial_fac = r_2 * self.k1 + (r_2**2) * self.k2 + (r_2**3) * self.k3 + (r_2**4) * self.k4
        radial = ((pts_ * fxy).T * radial_fac).T

        # tangential x = 2 p1 x' y' + p2 (r2 + 2 x'²)
        # tangential y = 2 p2 x' y' + p1 (r2 + 2 y'²)
        tangential = (
            2 * fxy * pts_ * np.flip(pts_, axis=1) * p1_p2 + np.flip(p1_p2) * fxy * (r_2 + 2 * np.power(pts_, 2).T).T
        )

        # x'' = x' *
        # u = fx * x'' + cx
        pts_distorted = pts_undistorted_pixel + radial + tangential

        return pts_distorted

    def _to_undistorted_points(self, pts_distorted_pixel: ArrayNx2) -> ArrayNx2:
        """Calculate undistorted points from distorted image pixel coordinates.

        :param pts_distorted_pixel: Distorted image pixel coordinates.
        :type pts_distorted_pixel: ArrayNx2
        :return: Undistorted image pixel coordinates.
        :rtype: ArrayNx2
        """

        # TODO There might be a problem with extremely high distortion values. Then the iteration by using the other
        #  direction as approx values will not work. Once I saw that than max_diff or pts_undistorted becomes np.nan

        # As the inversion is non-linear and can not be inverted one option is to use iteration
        # using the _to_distorted_point function to minimize the differences
        pts_undistorted = pts_distorted_pixel + (pts_distorted_pixel - self._to_distorted_points(pts_distorted_pixel))
        max_diff = np.max(np.linalg.norm(pts_distorted_pixel - self._to_distorted_points(pts_undistorted), axis=1))
        iteration = 0
        while max_diff > 0.001 and iteration < 30:
            iteration += 1
            pts_undistorted = pts_undistorted + (pts_distorted_pixel - self._to_distorted_points(pts_undistorted))
            max_diff = np.max(
                np.linalg.norm(
                    pts_distorted_pixel - self._to_distorted_points(pts_undistorted),
                    axis=1,
                )
            )

        return pts_undistorted
