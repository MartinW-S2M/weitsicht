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

import numpy as np
import pytest

from weitsicht.camera.opencv_perspective import CameraOpenCVPerspective


@pytest.fixture
def camera_class():
    cam_class = CameraOpenCVPerspective(
        width=5472,
        height=3648,
        fx=3753.0,
        fy=3753.0,
        cx=2721.2,
        cy=1813.7,
        k1=-0.0082,
        k2=0.00053,
        k3=0.0089,
        p1=-0.0015,
        p2=-0.00086,
    )
    assert int(cam_class.type)
    return cam_class


def test_generate_distortion_border(camera_class):
    """

    OpenCV is used as reference:
    import cv2
    import numpy as np
    img_points = np.array([[0., 0.], [912., 0.], [1824., 0.],
                       [2736., 0.], [3648., 0.], [4560., 0.],
                       [5472., 0.], [5472., 608.], [5472., 1216.],
                       [5472., 1824.], [5472., 2432.], [5472., 3040.],
                       [5472., 3648.], [4560., 3648.], [3648., 3648.],
                       [2736., 3648.], [1824., 3648.], [912., 3648.],
                       [0., 3648.], [0., 3040.], [0., 2432.],
                       [0., 1824.], [0., 1216.], [0., 608.]])
    # We need to shift as we generate our border at the pixel edge and opencv is defined at pixel center point
    img_points -= 0.5
    cm = np.array([[3753,0,2721.2],
                    [0,3753, 1813.7],
                    [0,0,1]])
    distCoeff = np.array([-0.0082, 0.00053, -0.0015, -0.00086, 0.0089])
    pts = cv2.undistortPoints(img_points, cm, distCoeff, P= cm)
    print(pts[:,0])

    """

    ref_unidostorted = np.array(
        [
            [3.6950856651737922, 4.926089041198338],
            [912.0498619053583, 1.166468531321243],
            [1824.2054469344891, 0.6661106176857174],
            [2736.258629429585, 0.21640851877396017],
            [3649.490791714201, -0.8701864992663104],
            [4565.120716488502, -1.8470306251776947],
            [5478.748812925189, 0.7278971900686884],
            [5481.816368398053, 607.5286790909195],
            [5483.398896310726, 1216.5151259677384],
            [5484.714312680481, 1826.6059687681761],
            [5486.112210815349, 2437.5928060337123],
            [5487.1874942863315, 3049.236019587731],
            [5486.611337234232, 3660.3181640614475],
            [4570.6074982813325, 3659.7546166435654],
            [3652.2821347336135, 3656.736160592074],
            [2736.32678114928, 3654.926858533194],
            [1821.5640557720176, 3655.111288891173],
            [906.7493917432471, 3656.5548025336248],
            [-3.949116069520187, 3655.8320076016635],
            [-5.12373219527899, 3046.095130475993],
            [-4.472306534030395, 2435.939819493362],
            [-3.2485788179346855, 1826.4876245091502],
            [-1.834510860566752, 1217.9239326802656],
            [0.09759060037094969, 610.405332026314],
        ]
    )

    # The ref_unidostorted is generate with opencv so we shift by 0.5
    ref_unidostorted += 0.5

    undistorted_border = camera_class._generate_distortion_border(points_between=5)
    print("Dist max[pix]:", np.max(undistorted_border - ref_unidostorted))
    assert np.allclose(undistorted_border, ref_unidostorted, atol=1e-3, rtol=0)


def test_pixel_image_to_camera_crs(camera_class):
    img_size = (camera_class.calibration_width, camera_class.calibration_height)
    prc_pt = camera_class.principal_point(img_size)

    # The principal point will not be distorted
    result = camera_class.pixel_image_to_camera_crs(prc_pt, img_size)
    assert result[0].tolist() == [0, 0, -1]


def test_pts_camara_crs_to_image_pixel(camera_class):
    img_size = (camera_class.calibration_width, camera_class.calibration_height)
    prc_pt = camera_class.principal_point(img_size)

    # The principal point will not be distorted
    result = camera_class.pts_camara_crs_to_image_pixel(np.array([0, 0, 1]), img_size)

    assert result[0].tolist() == prc_pt.tolist()


def test_distorted_to_undistorted(camera_class):
    img_size = (camera_class.calibration_width, camera_class.calibration_height)
    prc_pt = camera_class.principal_point(img_size)
    result = camera_class.distorted_to_undistorted(prc_pt, img_size)
    assert result[0].tolist() == prc_pt.tolist()
    test_points = np.array([[100, 100]])
    result = camera_class.distorted_to_undistorted(test_points, img_size)
    result_2 = camera_class.undistorted_to_distorted(result, img_size)
    assert np.linalg.norm(test_points - result_2) < 0.1


def test_undistorted_to_distorted(camera_class):
    img_size = (camera_class.calibration_width, camera_class.calibration_height)
    prc_pt = camera_class.principal_point(img_size)

    # The principal point will not be distorted
    result = camera_class.distorted_to_undistorted(prc_pt, img_size)

    assert result[0].tolist() == prc_pt.tolist()


def test_image_points_inside(camera_class):
    # Test with image points in camera calibration size
    image_points = np.array(
        [
            [
                0,
                0,
            ],
            [100, 100],
        ]
    )
    img_size = (camera_class.calibration_width, camera_class.calibration_height)
    valid_index = camera_class.undistorted_image_points_inside(image_points, image_size=img_size)

    assert not bool(valid_index[0])
    assert bool(valid_index[1])

    # Test with image size
    # these points are from opencv
    image_points = np.array([[2736.32678114928, 3654.926858533194], [2736.32678114928, 3654.926858533194]])
    image_points += 0.5

    # shift a little to be sure one is inside and one outside.
    # Undisotort iteration is about 1e-2 accurate min in the current configuration
    image_points = image_points + np.array([[0.05, 0.05], [-0.05, -0.05]])

    image_points = image_points / 2.0
    # Simulate image which is 50% of original image used for calibration
    img_size = (
        camera_class.calibration_width / 2.0,
        camera_class.calibration_height / 2.0,
    )
    valid_index = camera_class.undistorted_image_points_inside(image_points, image_size=img_size)

    assert not bool(valid_index[0])
    assert bool(valid_index[1])

    image_points = np.array(
        [
            [3.6950856651737922, 4.926089041198338],
            [3.6950856651737922, 4.926089041198338],
        ]
    )
    # The values are from opencv so we shift 0.5
    image_points += 0.5

    # We shift small pixel values. One is outside the other inside
    image_points = image_points + np.array([[0.01, 0.01], [-0.1, 0.01]])
    img_size = (camera_class.calibration_width, camera_class.calibration_height)
    valid_index = camera_class.undistorted_image_points_inside(image_points, image_size=img_size)

    assert bool(valid_index[0])
    assert not bool(valid_index[1])
