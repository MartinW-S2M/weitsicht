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
from pyproj import CRS

from weitsicht import Issue
from weitsicht.mapping.horizontal_plane import MappingHorizontalPlane
from weitsicht.transform.coordinates_transformer import CoordinateTransformer


def test_from_dict():
    mapper = MappingHorizontalPlane.from_dict({"type": "horizontalPlane", "plane_altitude": 43, "crs": "EPSG:4326+3855"})
    assert mapper is not None

    with pytest.raises(KeyError):
        MappingHorizontalPlane.from_dict({"type_key_missing": "whatsoever"})

    with pytest.raises(ValueError):
        MappingHorizontalPlane.from_dict({"type": "horizontalPlaneWrong"})


def test_param_dict():

    mapper = MappingHorizontalPlane.from_dict({"type": "horizontalPlane", "plane_altitude": 43.0, "crs": "EPSG:4979"})

    param = mapper.param_dict
    assert param == {
        "type": mapper.type.fullname,
        "plane_altitude": 43.0,
        "crs": mapper.crs.to_wkt() if mapper.crs is not None else None,
    }

    assert mapper.crs is not None
    assert mapper.crs.equals(CRS("EPSG:4979"))


def test_map_coordinates_from_rays():
    mapper = MappingHorizontalPlane.from_dict({"type": "horizontalPlane", "plane_altitude": 43.0, "crs": "EPSG:4979"})
    mapping_result = mapper.map_coordinates_from_rays(
        ray_start_crs_s=np.array([0, 0, 100]),
        ray_vectors_crs_s=np.array([0, 0, -1]),
        crs_s=CRS("EPSG:25833").to_3d(),
    )
    assert mapping_result.ok is True
    assert np.linalg.norm(mapping_result.coordinates - np.array([0, 0, 43])) < 0.0000001

    # test wrong direction

    mapping_result = mapper.map_coordinates_from_rays(
        ray_start_crs_s=np.array([0, 0, 100]),
        ray_vectors_crs_s=np.array([0, 0, 1]),
        crs_s=CRS("EPSG:25833"),
    )
    assert mapping_result.ok is False

    # Two rays, one fails one works
    mapping_result = mapper.map_coordinates_from_rays(
        ray_start_crs_s=np.array([[0, 0, 100], [0, 0, 100]]),
        ray_vectors_crs_s=np.array([[0, 0, 1], [0, 0, -1]]),
        crs_s=CRS("EPSG:25833"),
    )
    assert mapping_result.ok is True
    assert bool(np.any(np.isnan(mapping_result.coordinates[0]))) is True
    assert bool(np.any(np.isnan(mapping_result.coordinates[1]))) is False
    assert Issue.NO_INTERSECTION in mapping_result.issues

    # test parallel line
    mapping_result = mapper.map_coordinates_from_rays(
        ray_start_crs_s=np.array([0, 0, 100]),
        ray_vectors_crs_s=np.array([0.3, 0.4, 0.0]),
        crs_s=CRS("EPSG:25833"),
    )
    assert mapping_result.ok is False

    # Test different sizes of start and ray vector
    with pytest.raises(ValueError):
        mapper.map_coordinates_from_rays(
            ray_start_crs_s=np.array([[0, 0, 150], [0, 0, 100]]),
            ray_vectors_crs_s=np.array([0, 0, -1]),
            crs_s=CRS("EPSG:25833"),
        )


def test_map_heights_from_coordinates():

    # Different CRS and 3d coordinates used
    mapper = MappingHorizontalPlane.from_dict({"type": "horizontalPlane", "plane_altitude": 43.0, "crs": "EPSG:4979"})

    coordinates = np.array([602013.0, 5340384.696, 400.0])
    mapping_result = mapper.map_heights_from_coordinates(coordinates_crs_s=coordinates, crs_s=CRS("EPSG:25833"))
    assert mapping_result.ok is True
    assert np.linalg.norm(mapping_result.coordinates - np.array([602013.0, 5340384.696, 43.0])) < 0.00001

    # Same CRS and only 2D coordinates are used as input
    mapper = MappingHorizontalPlane.from_dict({"type": "horizontalPlane", "plane_altitude": 43.0, "crs": "EPSG:25833"})

    coordinates = np.array([602013.0, 5340384.696])
    mapping_result = mapper.map_heights_from_coordinates(coordinates_crs_s=coordinates, crs_s=CRS("EPSG:25833"))

    assert mapping_result.ok is True
    assert np.linalg.norm(mapping_result.coordinates - np.array([602013.0, 5340384.696, 43.0])) < 0.00001


def test_map_heights_from_coordinates_normals():

    # Different CRS and 3d coordinates used
    mapper = MappingHorizontalPlane(plane_altitude=400, crs=CRS(25833).to_3d())

    coordinates_src = np.array([4086085.57226463, 1200516.4908149, 4732729.71533306])
    mapping_result = mapper.map_heights_from_coordinates(coordinates_crs_s=coordinates_src, crs_s=CRS("EPSG:4978"))
    assert mapping_result.ok is True
    np.testing.assert_almost_equal(np.linalg.norm(mapping_result.coordinates - coordinates_src), 100)
    up_point = mapping_result.coordinates + mapping_result.normals * 100
    np.testing.assert_almost_equal(np.linalg.norm(up_point - coordinates_src), 0)


def test_map_rays_normals():

    # Different CRS and 3d coordinates used
    mapper = MappingHorizontalPlane(plane_altitude=400, crs=CRS(25833).to_3d())

    # ray more or less defined from UTM

    coordinates_src = np.array([4086085.57226463, 1200516.4908149, 4732729.71533306])
    vector = -np.array([0.63939621, 0.18785845, 0.74557474])  # test_map_heights_from_coordinates_normals
    # -> so invert and change a little to go down
    vector += np.array([0.2, -0.1, 0])
    mapping_result = mapper.map_coordinates_from_rays(
        ray_start_crs_s=coordinates_src, ray_vectors_crs_s=vector, crs_s=CRS("EPSG:4978")
    )
    assert mapping_result.ok is True
    np.testing.assert_almost_equal(mapping_result.normals, np.array([[0.63939621, 0.18785845, 0.74557474]]), decimal=4)

    c = CoordinateTransformer.from_crs(CRS(4978), CRS(25833).to_3d())
    assert c is not None
    should_be_vertical = c.transform_vector(mapping_result.coordinates, mapping_result.normals)
    np.testing.assert_almost_equal(should_be_vertical, np.array([[0, 0, 1]]))
