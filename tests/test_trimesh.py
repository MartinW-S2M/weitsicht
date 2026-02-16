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

from pathlib import Path

import numpy as np
import pytest
from pyproj import CRS

from weitsicht.exceptions import CRSInputError, CRSnoZaxisError, MappingError
from weitsicht.mapping.base_class import MappingType
from weitsicht.mapping.map_trimesh import MappingTrimesh

DATA_DIR = Path(__file__).parent.resolve() / "data"


@pytest.fixture
def mesh():

    # First test wrong datafiles:
    # Test wrong path
    with pytest.raises(FileNotFoundError):
        MappingTrimesh(DATA_DIR / "wrong_decimated_160k_31256.ply")

    # Test non supported file by rasterio
    with pytest.raises(MappingError):
        MappingTrimesh(__file__)

    # Test CRS system without z Axis

    trimesh_mapper = MappingTrimesh(DATA_DIR / "decimated_160k_31256.ply", crs=CRS("EPSG:31256+5778"))

    assert int(trimesh_mapper.type)
    # CRS.proj_crs_get_coordinate_system

    return trimesh_mapper


def test_type(mesh: MappingTrimesh):
    assert mesh.type == MappingType.Trimesh


def test_from_dict():
    mesh_path = DATA_DIR / "decimated_160k_31256.ply"
    crs = CRS(31256)

    # Test "type" key missing
    with pytest.raises(KeyError):
        dict_test = {
            "type_wrong": MappingType.Trimesh.fullname,
            "mesh_filepath": mesh_path.as_posix(),
            "crs": crs.to_wkt(),
            "coordinate_shift": [0, 0, 0],
        }
        MappingTrimesh.from_dict(dict_test)

    # Test wrong "type" string
    with pytest.raises(ValueError):
        dict_test = {
            "type": "wrong",
            "mesh_filepath": mesh_path,
            "crs": crs.to_wkt(),
            "coordinate_shift": [0, 0, 0],
        }
        MappingTrimesh.from_dict(dict_test)

    # Test "mesh_filepath" key missing
    with pytest.raises(KeyError):
        dict_test = {
            "type": MappingType.Trimesh.fullname,
            "crs": crs.to_wkt(),
            "coordinate_shift": [0, 0, 0],
        }
        MappingTrimesh.from_dict(dict_test)

    # Test that CRS has no z Axis defined
    with pytest.raises(CRSnoZaxisError):
        dict_test = {
            "type": MappingType.Trimesh.fullname,
            "mesh_filepath": mesh_path,
            "crs": crs.to_wkt(),
            "coordinate_shift": [0, 0, 0],
        }
        MappingTrimesh.from_dict(dict_test)

    # Test "crs" can not be loaded
    with pytest.raises(CRSInputError):
        dict_test = {
            "type": MappingType.Trimesh.fullname,
            "mesh_filepath": mesh_path,
            "crs": "asfdasf",
            "coordinate_shift": [0, 0, 0],
        }
        MappingTrimesh.from_dict(dict_test)

    dict_test = {
        "type": MappingType.Trimesh.fullname,
        "mesh_filepath": mesh_path,
        "crs": CRS("EPSG:31256+5778").to_wkt(),
        "coordinate_shift": None,
    }

    _ = MappingTrimesh.from_dict(dict_test)


def test_param_dict(mesh: MappingTrimesh):
    mesh_path = DATA_DIR / "decimated_160k_31256.ply"
    crs = CRS("EPSG:31256+5778")

    mesh_dict = mesh.param_dict
    dict_test = {
        "type": MappingType.Trimesh.fullname,
        "mesh_filepath": mesh_path.as_posix(),
        "crs": crs.to_wkt(),
        "coordinate_shift": None,
    }

    assert dict_test == mesh_dict

    mesh.translation = np.array([0.0, 200.0, 0.0])
    mesh_dict = mesh.param_dict

    dict_test = {
        "type": MappingType.Trimesh.fullname,
        "mesh_filepath": mesh_path.as_posix(),
        "crs": crs.to_wkt(),
        "coordinate_shift": [0.0, 200.0, 0.0],
    }

    assert dict_test == mesh_dict


def test_map_coordinates_from_rays(mesh: MappingTrimesh):
    ray_start = np.array([[2643.05, 342715.09, 4.95], [2643.20, 342717.07, 4.91]])

    ray_vector = np.array([[1, 0, 0], [1, 0, 0]])

    result = mesh.map_coordinates_from_rays(ray_start_crs_s=ray_start, ray_vectors_crs_s=ray_vector, crs_s=mesh.crs)
    assert result.ok is False

    ray_vector = np.array([[-1, 0, 0], [-1, 0, 0]])
    result = mesh.map_coordinates_from_rays(ray_start_crs_s=ray_start, ray_vectors_crs_s=ray_vector, crs_s=mesh.crs)

    assert result.ok is True


def test_map_heights_from_coordinates(mesh: MappingTrimesh):

    # The second coordinate is on a hole on the mesh
    # Therefore it should have the mask False for the second coordinate
    coordinates_to_check = np.array([[2638.8261, 342718.072, 100], [2639.06, 342719.40295, 100]])

    # Hand picked that coordinates
    height_ref = np.array([3.3225, 3.37239])

    result = mesh.map_heights_from_coordinates(coordinates_crs_s=coordinates_to_check, crs_s=mesh.crs)

    assert result.ok is True
    assert result.coordinates.shape == (2, 3)
    np.testing.assert_array_equal(result.mask, np.array([True, False]))

    # This coordinates work well
    coordinates_to_check = np.array([[2638.8261, 342718.072, 100], [2639.047977, 342719.404663, 100]])
    result = mesh.map_heights_from_coordinates(coordinates_crs_s=coordinates_to_check, crs_s=mesh.crs)

    assert result.ok is True
    assert result.coordinates.shape == (2, 3)
    assert np.all(result.mask)
    if np.max(np.abs(result.coordinates[:, 2] - height_ref)) > 0.001:
        raise AssertionError()
