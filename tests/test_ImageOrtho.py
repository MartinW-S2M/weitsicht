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

"""Test of ortho photos"""

from pathlib import Path

import numpy as np
import pyproj.network
import pytest
from pyproj import CRS

from weitsicht.image.base_class import ImageType
from weitsicht.image.ortho import ImageOrtho
from weitsicht.mapping.raster import MappingRaster
from weitsicht.utils import Issue

DATA_DIR = Path(__file__).parent.resolve() / "data"
pyproj.network.set_network_enabled(True)  # type: ignore


@pytest.fixture
def image():
    image = ImageOrtho.from_file(DATA_DIR / "44_2_op_2023_30cm.tif", CRS("EPSG:31256+5778"))

    assert image is not None
    assert int(image.type)

    assert image.position_wgs84 is not None
    assert image.position_wgs84_geojson is not None

    assert image.crs_wkt is not None
    assert image.crs_proj4 is not None

    image.mapper = MappingRaster(DATA_DIR / "44_2_dgm.tif", crs=CRS("EPSG:31256+8881"))

    assert np.isclose(image.resolution, 0.3, atol=0.001, rtol=0)

    assert image.geo_transform is not None

    return image


def test_type(image):
    assert image.type == ImageType.Orthophoto


def test_from_dict():

    crs = CRS("EPSG:31256+5778")
    dict_load = {
        "type": "ortho",
        "width": 1667,
        "height": 1667,
        "geo_transform": {
            "a": 0.29994001199760045,
            "b": 0.0,
            "c": -1750.0,
            "d": 0.0,
            "e": -0.29994001199760045,
            "f": 338800.0,
            "g": 0.0,
            "h": 0.0,
            "i": 1.0,
        },
        "resolution": 0.29994001199760045,
        "crs": crs.to_wkt(),
    }

    image_from_dict = ImageOrtho.from_dict(dict_load)
    coo = np.array(
        [
            [-1529.31, 338396.05, 34.7672685000015],
            [-1500.38, 338385.504, 34.7062945599976],
        ]
    )
    pix = np.array([[735.78046, 1346.7693], [832.23308, 1381.929664]])

    assert image_from_dict is not None
    image_from_dict.mapper = MappingRaster(DATA_DIR / "44_2_dgm.tif", crs=CRS("EPSG:31256+8881"))

    result = image_from_dict.map_points(pix)

    # Vienna height to GHA is a simple shift, thus, we are also testing the transformer here
    coo_gha = coo + np.array([0, 0, 156.68])
    assert result.ok is True
    assert np.allclose(result.coordinates, coo_gha, atol=1e-7, rtol=0)


def test_param_dict(image):
    test_dict = {
        "type": "ortho",
        "width": 1667,
        "height": 1667,
        "geo_transform": {
            "a": 0.29994001199760045,
            "b": 0.0,
            "c": -1750.0,
            "d": 0.0,
            "e": -0.29994001199760045,
            "f": 338800.0,
            "g": 0.0,
            "h": 0.0,
            "i": 1.0,
        },
        "resolution": 0.29994001199760045,
        "crs": image.crs_wkt,
    }

    assert image.param_dict == test_dict


def test_center(image):
    assert image.center == (833.5, 833.5)


def test_is_geo_referenced(image):
    assert image.is_geo_referenced


def test_image_points_inside(image):
    assert np.all(image.image_points_inside(np.array([[0, 0], [1200, 1200]])))
    assert not np.all(image.image_points_inside(np.array([[-1, -1], [1667, 1200]])))


def test_position_to_crs(image):
    # For orthophoto the image is the center of image using the mapper if specified for height
    # otherwise its zero in the images coordinates

    # Using GDAL for the center
    # gdallocationinfo - r bilinear - geoloc 44_2_dgm.tif - 1.5000e+033.3855e+05
    # Value: 33.17

    result = image.position_to_crs(CRS("EPSG:31256+5778").to_3d())
    assert np.isclose(result[2], 33.17 + 156.68, atol=0.0001, rtol=0.0)

    # Test where trafo of mapper and code parts which use transformation
    result = image.position_to_crs(image.mapper.crs)
    assert np.isclose(result[2], 33.17, atol=0.0001, rtol=0.0)

    # Case no mapper is specified
    image.mapper = None
    result = image.position_to_crs("EPSG:31256+5778")
    assert np.isclose(result[2], 0, atol=0.0001, rtol=0.0)


def test_project(image):
    # gdallocationinfo -r bilinear -geoloc 44_2_op_2023_30cm.tif -1529.31 338396.05
    #   Location: (735.78046P,1346.76930000004L)
    # gdallocationinfo -r bilinear -geoloc 44_2_op_2023_30cm.tif -1500.38 338385.504
    #   Location: (832.23308P,1381.92966399994L)

    ref_gdal = np.array([[735.78046, 1346.76930000004], [832.23308, 1381.92966399994]])

    coo = np.array([[-1529.31, 338396.05], [-1500.38, 338385.504]])

    result = image.project(coo, image.crs)

    assert result.ok is True
    np.testing.assert_array_equal(result.mask, np.array([True, True]))
    assert np.allclose(result.pixels, ref_gdal, atol=1e-8, rtol=0)

    coo = np.array([[-1760, 338396.05], [-1500.38, 338809.504]])
    result = image.project(coo, image.crs)
    assert result.ok is False
    assert Issue.INVALID_PROJECTIIONS in result.issues

    coo = np.array(
        [
            [-1529.31, 338396.05],
            [-1760, 338396.05],
            [-1500.38, 338809.504],
            [-1500.38, 338385.504],
        ]
    )
    result = image.project(coo, image.crs)
    assert result.ok is True
    assert Issue.INVALID_PROJECTIIONS in result.issues
    np.testing.assert_array_equal(result.mask, np.array([True, False, False, True]))


def test_map_center_point(image):
    # Using GDAL for the center
    # gdallocationinfo - r bilinear - geoloc 44_2_dgm.tif - 1.5000e+033.3855e+05
    # Value: 33.17
    # that value is in vienna height. Shift by 156.68 to be in GHA heights of image

    result = image.map_center_point()
    assert result.ok is True
    assert np.isclose(result.coordinates[0][2], 33.17 + 156.68, atol=0.0001, rtol=0.0)


def test_map_footprint(image):

    result = image.map_footprint()
    assert result.ok is True
    assert np.isclose(result.area, 250000, atol=1e-1, rtol=0)
    assert np.isclose(result.gsd, 0.3, atol=1e-1, rtol=0)

    # gdallocationinfo -r bilinear -geoloc 44_2_dgm.tif  -1529.31 338800.0 -> 26.036
    # gdallocationinfo -r bilinear -geoloc 44_2_dgm.tif  -1750.0  338300.0 -> 35.982
    # gdallocationinfo -r bilinear -geoloc 44_2_dgm.tif  -1250.0  338300.0 -> 33.887
    # gdallocationinfo -r bilinear -geoloc 44_2_dgm.tif  -1250.0  338800.0 -> 33.517

    ref_border = np.array(
        [
            [-1750.0, 338800.0, 26.036],
            [-1750.0, 338300.0, 35.982],
            [-1250.0, 338300.0, 33.887],
            [-1250.0, 338800.0, 33.517],
        ]
    )
    # shift to gha
    ref_border = ref_border + np.array([0, 0, 156.68])

    assert np.allclose(ref_border, result.coordinates, atol=1e-6, rtol=0)


def test_map_points(image):
    # Heights from GDAL
    # gdallocationinfo -r bilinear -geoloc 44_2_dgm.tif  -1529.31 338396.05
    # Location: (231.19P, 414.450000000012L)
    # Value: 34.7672685000015

    # gdallocationinfo - r bilinear - geoloc 44_2_dgm.tif - 1500.38 338385.504
    # Location: (260.12P, 424.995999999985L)
    # Value: 34.7062945599976

    coo = np.array(
        [
            [-1529.31, 338396.05, 34.7672685000015],
            [-1500.38, 338385.504, 34.7062945599976],
        ]
    )
    pix = np.array([[735.78046, 1346.7693], [832.23308, 1381.929664]])

    result = image.map_points(pix)

    # Vienna height to GHA is a simple shift, thus we are also testing the transformer here
    coo_gha = coo + np.array([0, 0, 156.68])
    assert result.ok is True
    assert np.allclose(result.coordinates, coo_gha, atol=1e-7, rtol=0)

    # Test outside Orthophoto
    # but inside mapping Raster
    # Test that we are inside the mapping raster
    # Not sure why I have tested this
    # res = image.mapper.map_heights_from_coordinates(np.array([[-1760, 338396.05]]), image.mapper.crs)
    # assert res is not None

    pix = np.array([[0, 0], [-1, -1], [735.78046, 1346.7693], [832.23308, 1381.929664]])
    result = image.map_points(pix)
    assert result.ok is True
    np.testing.assert_array_equal(result.mask, np.array([True, True, True, True]))

    # TODO find a test where some points are outside the mapping raster to check that issues are returned
    # assert Issue.OUTSIDE_RASTER in result.issues
