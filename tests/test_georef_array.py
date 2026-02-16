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

import time
from pathlib import Path

import numpy as np
import pytest
from pyproj import CRS, network

from weitsicht.geometry.intersection_plane import intersection_plane_mat_operation
from weitsicht.mapping.raster import MappingRaster

network.set_network_enabled(True)  # type: ignore

FIXTURE_DIR = Path(__file__).parent.resolve() / "data"


@pytest.fixture
def raster_mapper():
    mapper_raster = MappingRaster(
        raster_path=FIXTURE_DIR / "dhm_at_lamb_10m_2018.tif",
        # raster_path=r"E:\03_australia\dhm_at_lamb_10m_2018.tif",
        crs=CRS("EPSG:31287+5778"),
    )
    limit = (
        551842.0000000000000000,
        426123.0000000000000000,
        561956.0000000000000000,
        433466.0000000000000000,
    )
    mapper_raster.load_window(limit)

    return mapper_raster


@pytest.fixture
def raster_mapper_full():
    mapper_raster = MappingRaster(
        raster_path=FIXTURE_DIR / "dhm_at_lamb_10m_2018.tif",
        # raster_path=r"E:\03_australia\dhm_at_lamb_10m_2018.tif",
        crs=CRS("EPSG:31287+5778"),
        preload_full_raster=True,
    )

    return mapper_raster


def test_map_coordinates_from_rays(raster_mapper):
    ray_crs = CRS("EPSG:31287+5778")  # CRS("EPSG:25833+5778")

    point_3d = np.array(
        [
            [557057, 429465.0, 1520 + 200],
            [557057, 429465.0, 1520 + 200],
            [557057 + 20, 429465.0 + 20, 1520 + 200],
            [557057 + 20, 429465.0 + 20, 1520 + 200],
            [551842.0 - 10, 433466.0 - 50, 1520 + 200],
            [
                551842.0 - 200,
                433466.0 - 50,
                1520 + 200,
            ],  # That point is starting from outside the raster
            [551842.0 - 10, 433466.0 - 50, 1200],
            [551842.0 + 10, 433466.0 - 50, 1130],
            [551842.0 - 10, 433466.0 - 50, 1130],
            [551842.0 + 10, 433466.0, 1720],
        ]
    )

    ray_vec = np.array(
        [
            [0.3, -0.4, -0.80],
            [0.1, -0.3, -1.0],
            [0.3, -0.4, -0.80],
            [0.1, -0.3, -1.0],
            [0.4, 0.0, -1.0],
            [0.4, 0.0, -1.0],
            [0.6, 0.0, -0.8],
            [0.0, -1.0, -0.001],
            [1.0, 0.0, -0.01],
            [0.0, -4.0, -1],
        ]
    )

    b = raster_mapper.georef_array.map_coordinates_from_rays(
        ray_vectors_crs_s=ray_vec, ray_start_crs_s=point_3d, crs_s=ray_crs
    )
    assert b.ok is True
    # We will test 2 things
    # (1) if the intersection point is correct
    # (2) if the bilinear interpolation of that coordinate is correct
    # (3) if that is correct than the intersection is for sure correct as this is on the line
    # and the bilinear interpolated height for that point is also correct
    interp_plane, valid_mas = intersection_plane_mat_operation(ray_vec, point_3d, b.coordinates, np.array([0, 0, 1]))
    assert np.allclose(interp_plane, b.coordinates, atol=1e-7, rtol=0)
    bilin_coo = raster_mapper.georef_array.map_heights_from_coordinates(b.coordinates[:, :2], raster_mapper.crs)
    assert bilin_coo.ok is True
    assert np.allclose(b.coordinates, bilin_coo.coordinates, atol=1e-7, rtol=0)

    # This will test different directions of rays and if the line grid cell intersection delivers all cells
    ray_vec = np.array(
        [
            [0.3, 0.1, -0.80],
            [-0.3, 0.1, -0.80],
            [-0.3, -0.1, -0.80],
            [0.3, -0.1, -0.80],
            [0.3, 0.4, -0.80],
        ]
    )
    point_3d = np.zeros(ray_vec.shape) + np.array([557057, 429465.0, 1520 + 200])

    b = raster_mapper.georef_array.map_coordinates_from_rays(
        ray_vectors_crs_s=ray_vec, ray_start_crs_s=point_3d, crs_s=ray_crs
    )
    assert b.ok is True
    interp_plane, valid_mas = intersection_plane_mat_operation(ray_vec, point_3d, b.coordinates, np.array([0, 0, 1]))
    assert np.allclose(interp_plane, b.coordinates, atol=1e-7, rtol=0)
    bilin_coo = raster_mapper.georef_array.map_heights_from_coordinates(b.coordinates[:, :2], raster_mapper.crs)
    assert np.allclose(b.coordinates, bilin_coo.coordinates, atol=1e-7, rtol=0)

    ray_vec = np.array(
        [
            [-0.3, -0.1, -0.80],
            [-0.3, -0.12, -0.80],
            [-0.3, -0.1, -0.80],
            [-0.3, -0.1, -0.80],
            [-0.4, -0.4, -0.80],  # That one is special for testing
        ]
    )
    point_3d = np.zeros(ray_vec.shape) + np.array([557057, 429465.0, 1520 + 200])

    b = raster_mapper.georef_array.map_coordinates_from_rays(
        ray_vectors_crs_s=ray_vec, ray_start_crs_s=point_3d, crs_s=ray_crs
    )
    interp_plane, valid_mas = intersection_plane_mat_operation(ray_vec, point_3d, b.coordinates, np.array([0, 0, 1]))
    assert np.allclose(interp_plane, b.coordinates, atol=1e-7, rtol=0)
    bilin_coo = raster_mapper.georef_array.map_heights_from_coordinates(b.coordinates[:, :2], raster_mapper.crs)
    assert np.allclose(b.coordinates, bilin_coo.coordinates, atol=1e-7, rtol=0)

    # horizontal rays on X
    point_3d = np.array([[551842.0 - 10, 433466.0 - 50, 1130], [551842.0 - 10, 433466.0 - 50, 1130]])
    # This gives an error -> Check
    ray_vec = np.array([[1.0, -0.1, 0.0], [1.0, 0.0, 0]])
    b = raster_mapper.georef_array.map_coordinates_from_rays(
        ray_vectors_crs_s=ray_vec, ray_start_crs_s=point_3d, crs_s=ray_crs
    )
    # That plane needs to be vertical, otherwise that horizontal ray is parallel
    interp_plane, valid_mas = intersection_plane_mat_operation(
        ray_vec, point_3d, b.coordinates, np.array([0.1, 0.0, 0.0])
    )
    assert np.allclose(interp_plane, b.coordinates, atol=1e-7, rtol=0)
    bilin_coo = raster_mapper.georef_array.map_heights_from_coordinates(b.coordinates[:, :2], raster_mapper.crs)
    assert np.allclose(b.coordinates, bilin_coo.coordinates, atol=1e-7, rtol=0)

    # horizontal rays on Y
    point_3d = np.array([[551842.0 + 10, 433466.0 - 50, 1130], [551842.0 + 10, 433466.0 - 50, 1130]])
    # This gives an error -> Check
    ray_vec = np.array([[0.0, -1.0, 0.0], [0.0, -1.0, 0]])
    b = raster_mapper.georef_array.map_coordinates_from_rays(
        ray_vectors_crs_s=ray_vec, ray_start_crs_s=point_3d, crs_s=ray_crs
    )

    # That plane needs to be vertical, otherwise that horizontal ray is parallel
    interp_plane, valid_mas = intersection_plane_mat_operation(ray_vec, point_3d, b.coordinates, np.array([0, -1.0, 0.0]))
    assert np.allclose(interp_plane, b.coordinates, atol=1e-7, rtol=0)
    bilin_coo = raster_mapper.georef_array.map_heights_from_coordinates(b.coordinates[:, :2], raster_mapper.crs)
    assert np.allclose(b.coordinates, bilin_coo.coordinates, atol=1e-7, rtol=0)


def test_speed(raster_mapper_full):
    ray_crs = CRS("EPSG:31287+5778")  # CRS("EPSG:25833+5778")
    a = time.time()
    ray_vec = np.random.random((1000, 3)) / 100.0 + np.array([0, 0, -1])
    point_3d = np.random.random((1000, 3)) / 10.0 + np.array([556782, 429523, 1600])

    b = raster_mapper_full.georef_array.map_coordinates_from_rays(
        ray_vectors_crs_s=ray_vec, ray_start_crs_s=point_3d, crs_s=ray_crs
    )
    print(time.time() - a)
    interp_plane, valid_mas = intersection_plane_mat_operation(ray_vec, point_3d, b.coordinates)
    assert np.allclose(interp_plane, b.coordinates, atol=1e-7, rtol=0)
    bilin_coo = raster_mapper_full.georef_array.map_heights_from_coordinates(b.coordinates[:, :2], ray_crs)

    assert np.allclose(b.coordinates, bilin_coo.coordinates, atol=1e-7, rtol=1e-7)


def test_map_heights_from_coordinates(raster_mapper_full):
    # Tested with gdal -> gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 556538.9876 429158.5652
    # Report:
    # Location: (469.512998180733P, 430.567498733493L)
    # Band 1:
    # Value: 1284.07670385579
    coo = np.array(
        [
            [556538.9876, 429158.5652, 1284.07670385579],
            [556538.9876, 429158.5652, 1284.07670385579],
        ]
    )
    result = raster_mapper_full.georef_array.map_heights_from_coordinates(coo, raster_mapper_full.crs)
    assert result.ok is True
    assert np.allclose(result.coordinates, coo, atol=0.01, rtol=0)

    # Different CRS
    coo_utm = np.array(
        [
            [531550.98696, 5287808.89400, 1289.12332],
            [531547.64402920, 5287818.13604243, 1284.07670385579],
        ]
    )
    crs = CRS("EPSG:25833+5778")

    result = raster_mapper_full.georef_array.map_heights_from_coordinates(coo_utm, crs)
    assert result.ok is True
    assert np.allclose(result.coordinates, coo_utm, atol=0.01, rtol=0)

    # Edge case on the upper limit -> width of raster = 1011

    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 561950.0 429158.5652
    # Location: (1010.40023729484P,430.567498733493L)
    #   Value: 1024.56193042452
    #   coordinates in raster crs
    coo = np.array([561950.0, 429158.5652])
    coo_map_heights = raster_mapper_full.georef_array.map_heights_from_coordinates(coo, raster_mapper_full.crs)
    assert coo_map_heights.ok is True
    assert np.allclose(coo_map_heights.coordinates[0, 2], 1024.56193042452, atol=1e-6, rtol=0)

    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 561955.0 429158.5652
    #   Location: (1010.90003954914P,430.567498733493L)
    #   Value: 1024.73956631826
    coo = np.array([561955.0, 429158.5652])
    coo_map_heights = raster_mapper_full.georef_array.map_heights_from_coordinates(coo, raster_mapper_full.crs)
    assert coo_map_heights.ok is True
    assert np.allclose(coo_map_heights.coordinates[0, 2], 1024.73956631826, atol=1e-6, rtol=0)

    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 561956.0 429158.5652
    #   Location: (1011P,430.567498733493L)
    # Location is off this file! No further details to report.
    coo = np.array([561956.0, 429158.5652])
    coo_map_heights = raster_mapper_full.georef_array.map_heights_from_coordinates(coo, raster_mapper_full.crs)
    assert coo_map_heights.ok is False

    # Edge case on the lower limit ->
    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 551842.0 429158.5652
    #   Location: (0P,430.567498733493L)
    #    Value: 952.09822577356
    # coordinates in raster crs
    coo = np.array([551842.0, 429158.5652])
    coo_map_heights = raster_mapper_full.georef_array.map_heights_from_coordinates(coo, raster_mapper_full.crs)
    assert coo_map_heights.ok is True
    assert np.allclose(coo_map_heights.coordinates[0, 2], 952.09822577356, atol=1e-6, rtol=0)

    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 551845.0 429158.5652
    # Report:
    #   Location: (0.299881352577358P,430.567498733493L)
    #   Band 1:
    #     Value: 952.09822577356
    coo = np.array([551845.0, 429158.5652])
    coo_map_heights = raster_mapper_full.georef_array.map_heights_from_coordinates(coo, raster_mapper_full.crs)
    assert coo_map_heights.ok is True
    assert np.allclose(coo_map_heights.coordinates[0, 2], 952.09822577356, atol=1e-6, rtol=0)

    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 551847.5 429158.5652
    # Report:
    #   Location: (0.549782479727583P,430.567498733493L)
    #   Band 1:
    #     Value: 952.275286000538
    coo = np.array([551847.5, 429158.5652])
    coo_map_heights = raster_mapper_full.georef_array.map_heights_from_coordinates(coo, raster_mapper_full.crs)
    assert coo_map_heights.ok is True
    assert np.allclose(coo_map_heights.coordinates[0, 2], 952.275286000538, atol=1e-6, rtol=0)

    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 551845.0 433464.5652
    #  Location: (0.299881352577358P,0.143421380911605L)
    #  Value: 1102.26696777344
    coo = np.array([551845.0, 433464.5652])
    coo_map_heights = raster_mapper_full.georef_array.map_heights_from_coordinates(coo, raster_mapper_full.crs)
    assert coo_map_heights.ok is True
    assert np.allclose(coo_map_heights.coordinates[0, 2], 1102.26696777344, atol=1e-6, rtol=0)


def test_map_heights_vertical_equal(raster_mapper_full):
    # Tested with gdal -> gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 556538.9876 429158.5652
    # Report:
    # Location: (469.512998180733P, 430.567498733493L)
    # Band 1:
    # Value: 1284.07670385579

    height_ref = 1284.07670385579

    # That test is for both crs are the same
    coo = np.array([[556538.9876, 429158.5652, 2000]])
    ray_vec = np.array([[0.0, 0.0, -1.0]])

    result_vertical_ray = raster_mapper_full.georef_array.map_coordinates_from_rays(
        ray_vectors_crs_s=ray_vec, ray_start_crs_s=coo, crs_s=raster_mapper_full.crs
    )

    result_map_heights = raster_mapper_full.georef_array.map_heights_from_coordinates(coo, raster_mapper_full.crs)

    assert result_map_heights.ok is True
    assert result_vertical_ray.ok is True
    assert np.allclose(
        result_vertical_ray.coordinates,
        result_map_heights.coordinates,
        atol=0.001,
        rtol=0,
    )
    assert np.isclose(height_ref, result_map_heights.coordinates[0, 2], atol=0.001, rtol=0)

    # That test is for different crs systems
    coo = np.array([[531547.64402920, 5287818.13604243, 2000]])
    crs = CRS("EPSG:25833+5778")
    ray_vec = np.array([[0.0, 0.0, -1.0]])

    result_vertical_ray = raster_mapper_full.georef_array.map_coordinates_from_rays(
        ray_vectors_crs_s=ray_vec, ray_start_crs_s=coo, crs_s=crs
    )

    result_map_heights = raster_mapper_full.georef_array.map_heights_from_coordinates(coo, crs)

    assert result_map_heights.ok is True
    assert result_vertical_ray.ok is True
    assert np.allclose(
        result_vertical_ray.coordinates,
        result_map_heights.coordinates,
        atol=0.001,
        rtol=0,
    )
    assert np.isclose(height_ref, result_vertical_ray.coordinates[0, 2], atol=0.01, rtol=0)

    # Edge case on the upper limit -> width of raster = 1011
    # These are valid position for the ray intersection
    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 561950.0 429158.5652
    # Location: (1010.40023729484P,430.567498733493L)
    #   Value: 1024.56193042452
    #   coordinates in raster crs
    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 551847.5 429158.5652
    #   Location: (0.549782479727583P,430.567498733493L)
    #    Value: 952.275286000538

    height_ref = np.array([1024.56193042, 952.275286000538])

    coo = np.array([[561950.0, 429158.5652, 8000], [551847.5, 429158.5652, 8000]])

    ray_vec = np.array([[0.0, 0.0, -1.0]])
    rays = np.zeros(coo.shape, dtype=float) + ray_vec

    result_map_heights = raster_mapper_full.georef_array.map_heights_from_coordinates(coo, raster_mapper_full.crs)
    result_vertical_ray = raster_mapper_full.georef_array.map_coordinates_from_rays(
        ray_vectors_crs_s=rays, ray_start_crs_s=coo, crs_s=raster_mapper_full.crs
    )
    assert result_map_heights.ok is True
    assert result_vertical_ray.ok is True
    assert np.allclose(
        result_vertical_ray.coordinates,
        result_map_heights.coordinates,
        atol=0.001,
        rtol=0,
    )
    assert np.allclose(height_ref, result_vertical_ray.coordinates[:, 2], atol=0.01, rtol=0)

    # This are vertical rays little bit inside the allowed border
    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 561950.9980217605 429158.5652
    #   Location: (1010.49999999999P,430.567498733493L)
    #   Value: 1024.73956631824
    coo = np.array([[561950.9980217605, 429158.5652, 6000]])
    ray_vec = np.array([[0.0, 0.0, -1.0]])

    result_vertical_ray = raster_mapper_full.georef_array.map_coordinates_from_rays(
        ray_vectors_crs_s=ray_vec, ray_start_crs_s=coo, crs_s=raster_mapper_full.crs
    )

    assert result_vertical_ray.ok is True
    assert np.allclose(1024.73956631824, result_vertical_ray.coordinates[:, 2], atol=1e-7, rtol=0)

    # This are vertical rays little bit outside the allowed border
    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 561950.9980217605 429158.5652
    #   Location: (1010.49999999999P,430.567498733493L)
    #   Value: 1024.73956631824
    coo = np.array([[561950.998021761, 429158.5652, 6000]])
    ray_vec = np.array([[0.0, 0.0, -1.0]])

    coo_map_heights = raster_mapper_full.georef_array.map_coordinates_from_rays(
        ray_vectors_crs_s=ray_vec, ray_start_crs_s=coo, crs_s=raster_mapper_full.crs
    )
    assert coo_map_heights.ok is False

    # This are vertical rays within the raster but to close to the border where
    # the bilinear interpolation of a ray is not working anymore as the gdal bilinear interpolation where
    # For pixle coordinates lower 0.5 just 0.5 is used and similar for the max values
    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 561955.0 429158.5652
    #   Location: (1010.90003954914P,430.567498733493L)
    #   Value: 1024.73956631826
    coo = np.array([[561955.0, 429158.5652, 6000]])
    ray_vec = np.array([[0.0, 0.0, -1.0]])
    raster_mapper_full.georef_array.map_coordinates_from_rays(
        ray_vectors_crs_s=ray_vec, ray_start_crs_s=coo, crs_s=raster_mapper_full.crs
    )
    assert coo_map_heights.ok is False

    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 551845.0 429158.5652
    #   Location: (0.299881352577358P,430.567498733493L)
    #   Value: 952.09822577356
    coo = np.array([[551845.0, 429158.5652, 6000]])
    raster_mapper_full.georef_array.map_coordinates_from_rays(
        ray_vectors_crs_s=ray_vec, ray_start_crs_s=coo, crs_s=raster_mapper_full.crs
    )
    assert coo_map_heights.ok is False

    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 551845.0 433464.5652
    #  Location: (0.299881352577358P,0.143421380911605L)
    #  Value: 1102.26696777344
    coo = np.array([[551845.0, 433464.5652, 6000]])

    raster_mapper_full.georef_array.map_coordinates_from_rays(
        ray_vectors_crs_s=ray_vec, ray_start_crs_s=coo, crs_s=raster_mapper_full.crs
    )
    assert coo_map_heights.ok is False
    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 551842.0 429158.5652
    #   Location: (0P,430.567498733493L)
    #    Value: 952.09822577356
    coo = np.array([[551842.0, 429158.5652, 6000]])
    raster_mapper_full.georef_array.map_coordinates_from_rays(
        ray_vectors_crs_s=ray_vec, ray_start_crs_s=coo, crs_s=raster_mapper_full.crs
    )
    assert coo_map_heights.ok is False
    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 561956.0 429158.5652
    #   Location: (1011P,430.567498733493L)
    # Location is off this file! No further details to report.
    coo = np.array([[561956.0, 429158.5652, 6000]])
    raster_mapper_full.georef_array.map_coordinates_from_rays(
        ray_vectors_crs_s=ray_vec, ray_start_crs_s=coo, crs_s=raster_mapper_full.crs
    )
    assert coo_map_heights.ok is False
