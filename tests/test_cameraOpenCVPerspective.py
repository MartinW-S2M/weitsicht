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

import numpy
import pytest

from weitsicht.camera.base_perspective import CameraType
from weitsicht.camera.opencv_perspective import CameraOpenCVPerspective


@pytest.fixture
def camera_class():
    return CameraOpenCVPerspective(
        width=5472,
        height=3648,
        fx=3753.0,
        fy=3754.0,
        cx=2721.2,
        cy=1813.7,
        k1=-0.082,
        k2=0.053,
        k3=0.0089,
        p1=-0.0015,
        p2=-0.00086,
    )


def test_camera_inti():

    with pytest.raises(ValueError):
        CameraOpenCVPerspective(width=0, height=3648, fx=3753.0, fy=3754.0)
    with pytest.raises(ValueError):
        CameraOpenCVPerspective(width=5472, height=0, fx=3753.0, fy=3754.0)
    with pytest.raises(ValueError):
        CameraOpenCVPerspective(width=5472, height=3648, fx=0.0, fy=3754.0)
    with pytest.raises(ValueError):
        CameraOpenCVPerspective(width=5472, height=3648, fx=3753.0, fy=0)

    camera = CameraOpenCVPerspective(width=5472, height=3648, fx=3753.0, fy=3753.0)
    assert camera.cx == 5472 / 2.0 - 0.5
    assert camera.cy == 3648 / 2.0 - 0.5


def test_from_dict(camera_class):
    cam = CameraOpenCVPerspective.from_dict(
        {
            "type": "OpenCV",
            "calib_width": 5472,
            "calib_height": 3648,
            "fx": 3753.0,
            "fy": 3754.0,
            "cx": 2721.2,
            "cy": 1813.7,
            "k1": -0.082,
            "k2": 0.053,
            "k3": 0.0089,
            "k4": 0.0,
            "p1": -0.0015,
            "p2": -0.00086,
        }
    )
    assert cam.param_dict == camera_class.param_dict

    with pytest.raises(KeyError):
        cam = CameraOpenCVPerspective.from_dict(
            {
                "type": "OpenCV",
                "calib_width": 5472,
                "calib_height": 3648,
                "fx1": 3753.0,
                "fy": 3754.0,
                "cx": 2721.2,
                "cy": 1813.7,
                "k1": -0.0082,
                "k2": 0.00053,
                "k3": 0.0089,
                "k4": 0.0,
                "p1": -0.0015,
                "p2": -0.00086,
            }
        )

    with pytest.raises(ValueError):
        cam = CameraOpenCVPerspective.from_dict(
            {
                "type": "OpenCV",
                "calib_width": 0,
                "calib_height": 3648,
                "fx": 3753.0,
                "fy": 3754.0,
                "cx": 2721.2,
                "cy": 1813.7,
                "k1": -0.0082,
                "k2": 0.00053,
                "k3": 0.0089,
                "k4": 0.0,
                "p1": -0.0015,
                "p2": -0.00086,
            }
        )


def test_type(camera_class):
    assert camera_class.type == CameraType.OpenCV


def test_param_dict(camera_class):
    assert camera_class.param_dict == {
        "type": "OpenCV",
        "calib_width": 5472,
        "calib_height": 3648,
        "fx": 3753.0,
        "fy": 3754.0,
        "cx": 2721.2,
        "cy": 1813.7,
        "k1": -0.082,
        "k2": 0.053,
        "k3": 0.0089,
        "k4": 0.0,
        "p1": -0.0015,
        "p2": -0.00086,
    }


def test_focal_length_for_gsd_in_pixel(camera_class):
    assert camera_class.focal_length_for_gsd_in_pixel == 3753.5


def test__principal_point(camera_class):
    assert camera_class._principal_point.tolist() == [camera_class.cx, camera_class.cy]


def test__camara_crs_to_image_crs(camera_class):
    result = camera_class._camara_crs_to_image_crs(numpy.array([0, 0, 1]))
    assert result[0].tolist() == [camera_class.cx, camera_class.cy]


def test__image_crs_to_vector_in_camera_crs(camera_class):
    result = camera_class._image_crs_to_vector_in_camera_crs(numpy.array([camera_class.cx, camera_class.cy]))
    assert result[0].tolist() == [0, 0, -1]


def test__to_distorted_points(camera_class):
    # The principal point will not be distorted
    result = camera_class._to_distorted_points(numpy.array([[camera_class.cx, camera_class.cy]]))

    assert result[0].tolist() == [camera_class.cx, camera_class.cy]


def test__to_undistorted_points(camera_class):
    result = camera_class._to_undistorted_points(numpy.array([[camera_class.cx, camera_class.cy]]))
    assert result[0].tolist() == [camera_class.cx, camera_class.cy]
    test_points = numpy.array([[10, 10]])
    result = camera_class._to_undistorted_points(test_points)
    result_2 = camera_class._to_distorted_points(result)
    assert numpy.linalg.norm(test_points - result_2) < 0.1
