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

"""Test of perspective image"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pyproj.network
import pytest
from pyproj import CRS

from weitsicht import CRSInputError, Issue, MapperMissingError, MappingBackendError, MappingError, NotGeoreferencedError
from weitsicht.camera.opencv_perspective import CameraOpenCVPerspective
from weitsicht.image.perspective import ImagePerspective
from weitsicht.mapping.base_class import MappingBase, MappingType
from weitsicht.mapping.horizontal_plane import MappingHorizontalPlane
from weitsicht.transform.coordinates_transformer import CoordinateTransformer
from weitsicht.transform.rotation import Rotation
from weitsicht.utils import ArrayNx2, ArrayNx3, MappingResult

pyproj.network.set_network_enabled(True)  # pyright: ignore[reportPrivateImportUsage]


class _TestMapperBase(MappingBase):
    @classmethod
    def from_dict(cls, mapper_dict: Mapping[str, Any]) -> "_TestMapperBase":
        raise NotImplementedError

    @property
    def type(self) -> MappingType:
        return MappingType.HorizontalPlane

    @property
    def param_dict(self) -> Mapping[str, Any]:
        return {"type": self.type.fullname}

    def map_coordinates_from_rays(
        self,
        ray_vectors_crs_s: ArrayNx3,
        ray_start_crs_s: ArrayNx3,
        crs_s: CRS | None = None,
        transformer: CoordinateTransformer | None = None,
    ) -> MappingResult:
        raise NotImplementedError

    def map_heights_from_coordinates(
        self,
        coordinates_crs_s: ArrayNx3 | ArrayNx2,
        crs_s: CRS | None = None,
        transformer: CoordinateTransformer | None = None,
    ) -> MappingResult:
        raise NotImplementedError


@pytest.fixture
def image():
    position = np.array([478397.4630808606, -2134438.8903628434, 149.8566026660398])
    rotation = Rotation.from_opk_degree(1.4890740840939083, -0.0032084676638993266, 78.52862383108489)
    camera = CameraOpenCVPerspective(
        width=5472,
        height=3648,
        fx=3.75303332732689296e03,
        fy=3.75303332732689296e03,
        cx=2.72122638969748232e03,
        cy=1.81369176603311416e03,
        k1=-0.00825592,
        k2=0.00053454,
        k3=0.00894906,
        p1=-0.00150822,
        p2=-0.00086861,
    )

    image = ImagePerspective(width=5472, height=3648, camera=camera, position=position, orientation=rotation)

    assert int(image.type)

    assert image._position is not None
    assert image.position_wgs84 is None
    assert image.position_wgs84_geojson is None

    assert image.crs_wkt is None
    assert image.crs_proj4 is None

    # For testing set crs after creation to test crs.setter
    image.crs = CRS("EPSG:32655+3855")

    assert image.position_wgs84
    assert image.position_wgs84_geojson

    assert image.crs_wkt
    assert image.crs_proj4

    return image


def test_project_simple_eor(image):
    """
    As reference OpenCV is used
    Examples:
    .. code-block:: python3

        import cv2
        import numpy as np
        cm = np.array([[3.75303332732689296e+03,0,2.72122638969748232e+03],
                      [0,3.75303332732689296e+03, 1.81369176603311416e+03],
                      [0,0,1]])
        adp = np.array([-0.00825592,0.00053454, -0.00150822, -0.00086861, 0.00894906])
        obj = np.array([[50.0, -4.0, -80.0],
                                          [-36.0, 23.5, -86.5]])
        res = cv2.projectPoints(obj,
                                np.array([np.pi, 0.0, 0.0]),  # rvec
                                np.array([0.0, 0.0, 0.0]),  # tvec
                                cm, adp)
        print(res[0][:,0])
            array([[5056.5461599 , 1998.39462748],
                   [1158.97950782,  793.01915317]])
    """

    # We need to shift opencv's results by 0.5 to be in our image coordinate system in the top left corner
    reference_opencv = np.array([[5056.5461599, 1998.39462748], [1158.97950782, 793.01915317]])
    reference_opencv += 0.5

    # we change the image eor to fit that example without rotation
    image._position = np.array([0, 0, 0])
    image._orientation = Rotation(np.eye(3, 3))

    obj = np.array([[50.0, -4.0, -80.0], [-36.0, 23.5, -86.5]])

    res = image.project(obj, image.crs)

    if res.ok is False:
        raise AssertionError()
    print("Max diff[px]: ", np.max(reference_opencv - res.pixels))
    assert np.allclose(reference_opencv, res.pixels, atol=1e-6, rtol=0)


def test_from_dict():
    cam = {
        "type": "OpenCV",
        "calib_width": 1000,
        "calib_height": 600,
        "fx": 1000.0,
        "fy": 1000,
    }
    dict_load = {
        "type": "perspective",
        "width": 0,
        "height": 600,
        "position": [1000.0, 2000.0, 0.0],
        "orientation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "camera": cam,
    }

    with pytest.raises(ValueError):
        ImagePerspective.from_dict(dict_load)

    dict_load = {
        "type": "perspective",
        "width": 1000,
        "height": 600,
        "position": [1000.0, 2000.0, 0.0],
        "orientation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "camera": cam,
    }
    image = ImagePerspective.from_dict(dict_load)
    assert image is not None


def test_param_dict(image):

    dict_return = image.param_dict
    assert dict_return["type"] == "perspective"


def test_project(image):
    """
    As reference OpenCV is used
    Examples:
    .. code-block:: python3

        # generate the object points
        mapper = MappingHorizontalPlane(crs=image.crs,plane_altitude=-100)
        obj = []
        img = []
        for x in range(10):
            img.append(np.random.random(2) * np.array([image.width, image.height]), )
            mapper.plane_altitude = image._position[2] -100 + np.random.random(1) * 50
            obj.append(image.map_points(img[-1],mapper)[0][0])
        img = np.array(img)
        obj = np.array(obj)

        import cv2
        import numpy as np
        cm = np.array([[3.75303332732689296e+03,0,2.72122638969748232e+03],
                      [0,3.75303332732689296e+03, 1.81369176603311416e+03],
                      [0,0,1]])
        adp = np.array([-0.00825592,0.00053454, -0.00150822, -0.00086861, 0.00894906])
        r = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) @ image._orientation.matrix.T
        # we need to shift opencv coordinates, otherwise it is not working for me
        # so we use obj-image._position and for prc we use [0,0,0]
        res = cv2.projectPoints(obj-image._position,
                                cv2.Rodrigues(r)[0],  # rvec
                                np.array([0.0, 0.0, 0.0]),  # tvec
                                cm, adp)
        print(res[0][:,0])
    """

    # We need to shift opencv's results by 0.5 to be in our image coordinate system in the top left corner
    reference_projections_opencv = np.array(
        [
            [2688.64576446, 957.78871189],
            [3510.25906442, 3232.55771521],
            [2845.19962912, 2291.01180443],
            [5386.00655929, 120.081348],
            [2776.50009998, 1101.49116565],
            [1847.13936088, 2504.56508033],
            [1980.88286791, 3277.45229158],
            [1315.17429357, 1884.93520377],
            [4367.11953191, 2365.94925465],
            [2600.55420857, 2992.18744232],
        ]
    )
    reference_projections_opencv += 0.5

    reference_point_3d = np.array(
        [
            [478385.3398438, -2134435.50480354, 96.05025393],
            [478433.07978594, -2134425.35613293, 64.13457151],
            [478410.26066863, -2134435.66823291, 52.49376194],
            [478378.17474396, -2134386.67264452, 87.00149474],
            [478383.77581348, -2134433.04157406, 75.14200689],
            [478407.53563241, -2134456.79870152, 74.42527293],
            [478430.02680608, -2134462.10315249, 54.60948133],
            [478393.51629712, -2134463.35600116, 78.27010139],
            [478412.86446323, -2134410.56283285, 84.33134482],
            [478414.56715098, -2134442.74333801, 93.11737895],
        ]
    )

    res = image.project(reference_point_3d, image.crs)

    if res.ok is False:
        raise AssertionError()
    # Within the direction of 3d to image as openCV is defined we are accurate to 1e-9 pixel
    print("Max diff[px]: ", np.max(reference_projections_opencv - res.pixels))
    assert np.allclose(reference_projections_opencv, res.pixels, atol=1e-8, rtol=0)


def test_map_footprint(image: ImagePerspective):
    mapper = MappingHorizontalPlane(0, image.crs)

    image.map_footprint(mapper=mapper)
    image.map_footprint(mapper=mapper, points_per_edge=5)


def test_map_geometry(image: ImagePerspective):
    # Point in 3d which is the reference
    reference_point_3d = np.array(
        [
            [478410.26066863, -2134435.66823291, 52.49376194],
            [478412.86446323, -2134410.56283285, 52.49376194],
        ]
    )
    point_crs = CRS("EPSG:32655+3855")
    # Reference Projection from OpencV projectPoints of 3d reference
    """
        .. code-block:: python3

            import cv2
            import numpy as np
            cm = np.array([[3.75303332732689296e+03,0,2.72122638969748232e+03],
                          [0,3.75303332732689296e+03, 1.81369176603311416e+03],
                          [0,0,1]])
            adp = np.array([-0.00825592,0.00053454, -0.00150822, -0.00086861, 0.00894906])
            r = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) @ image._orientation.matrix.T
            # we need to shift opencv coordinates, otherwise it is not working for me
            # so we use obj-image._position and for prc we use [0,0,0]
            res = cv2.projectPoints(reference_point_3d-image._position,
                                    cv2.Rodrigues(r)[0],  # rvec
                                    np.array([0.0, 0.0, 0.0]),  # tvec
                                    cm, adp)
            print(res[0][:,0])

    """
    # We use OpenCV to project 3d points. These projections are used for mapping than
    reference_projection = np.array([[2845.19962912, 2291.01180443], [3803.58732855, 2193.72096208]])
    # OpenCV has its origin in the center pixel of the top left pixel
    reference_projection += 0.5

    # We use the 3d point to get the height for our plane mapper
    mapper = MappingHorizontalPlane(crs=point_crs, plane_altitude=reference_point_3d[0, 2])
    image.mapper = mapper
    mapping_result = image.map_points(reference_projection)

    # The mapped point will be in the image coordinate system so we transform back to the reference point crs

    if mapping_result.ok is False:
        raise AssertionError()

    coo_trafo = CoordinateTransformer.from_crs(image.crs, point_crs)
    if coo_trafo is not None:
        mapped_point_pt_crs = coo_trafo.transform(mapping_result.coordinates)
    else:
        mapped_point_pt_crs = mapping_result.coordinates * 1.0

    print("Max coo diff[m]:", np.max(np.abs(mapped_point_pt_crs - reference_point_3d)))
    assert np.allclose(mapped_point_pt_crs, reference_point_3d, atol=1e-4, rtol=0)


def test_map_points(image):

    image.mapper = None

    with pytest.raises(MapperMissingError):
        res = image.map_points(np.array([0, 2]))

    mapper = MappingHorizontalPlane(5000, crs=CRS("EPSG:32655"))
    res = image.map_points(np.array([0, 2]), mapper=mapper)
    assert res.ok is False
    assert Issue.NO_INTERSECTION in res.issues

    # Test position afterwards for coverage
    image._position = None
    with pytest.raises(NotGeoreferencedError):
        res = image.map_points(np.array([0, 2]))


def test_map_points_wraps_foreign_mapper_errors(image: ImagePerspective):
    class ExplodingMapper(_TestMapperBase):
        def map_coordinates_from_rays(
            self,
            ray_vectors_crs_s: ArrayNx3,
            ray_start_crs_s: ArrayNx3,
            crs_s: CRS | None = None,
            transformer: CoordinateTransformer | None = None,
        ) -> MappingResult:
            raise RuntimeError("boom")

    with pytest.raises(MappingBackendError):
        image.map_points(np.array([0, 2]), mapper=ExplodingMapper())


def test_map_points_does_not_wrap_mapping_errors(image: ImagePerspective):
    class MappingErrorMapper(_TestMapperBase):
        def map_coordinates_from_rays(
            self,
            ray_vectors_crs_s: ArrayNx3,
            ray_start_crs_s: ArrayNx3,
            crs_s: CRS | None = None,
            transformer: CoordinateTransformer | None = None,
        ) -> MappingResult:
            raise MappingError("expected mapping error")

    with pytest.raises(MappingError):
        image.map_points(np.array([0, 2]), mapper=MappingErrorMapper())


def test_map_points_does_not_wrap_crs_input_errors(image: ImagePerspective):
    class CRSInputErrorMapper(_TestMapperBase):
        def map_coordinates_from_rays(
            self,
            ray_vectors_crs_s: ArrayNx3,
            ray_start_crs_s: ArrayNx3,
            crs_s: CRS | None = None,
            transformer: CoordinateTransformer | None = None,
        ) -> MappingResult:
            raise CRSInputError("invalid CRS input")

    with pytest.raises(CRSInputError):
        image.map_points(np.array([0, 2]), mapper=CRSInputErrorMapper())


def test_project_and_map_geometry(image: ImagePerspective):
    """This will test a lot of image functions"""

    # Picked point in image which corresponds to reference point in 3d
    mapper = MappingHorizontalPlane(crs=image.crs, plane_altitude=-100)

    assert image._position is not None
    obj = []
    img = []
    for _ in range(10):
        img.append(
            np.random.random(2) * np.array([image.width, image.height]),
        )
        mapper.plane_altitude = image._position[2] - 100 + np.random.random(1) * 50
        mapping_result = image.map_points(img[-1], mapper)
        assert mapping_result.ok is True

        obj.append(mapping_result.coordinates[0])

    img = np.array(img)
    obj = np.array(obj)

    res = image.project(obj, image.crs)

    # The difference is depending on the tolerance we stop the iteration of the undistorting during mapping
    # If we would make more iterations it would be better than 1e-12
    # But we set the tolerance lower at the iteration to waste not time in pixel dimension of < 0.01
    assert res.ok is True
    print("Max diff[px]: ", np.max(img - res.pixels))
    assert np.allclose(img, res.pixels, atol=1e-2, rtol=0)


def test_pixel_to_ray_vector(image):
    reference_projection = np.array([[2845.19962912, 2291.01180443], [3803.58732855, 2193.72096208]])

    reference_point_3d = np.array(
        [
            [478410.26066863, -2134435.66823291, 52.49376194],
            [478412.86446323, -2134410.56283285, 52.49376194],
        ]
    )

    line_vec = reference_point_3d - image._position
    vec_reference = np.divide(line_vec.T, np.linalg.norm(line_vec, axis=1)).T
    vec = image.pixel_to_ray_vector(reference_projection)

    assert np.allclose(vec, vec_reference, rtol=0, atol=0.001)
