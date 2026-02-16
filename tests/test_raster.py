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
from pyproj import network as network_pyproj

from weitsicht.exceptions import CRSnoZaxisError, MappingError
from weitsicht.mapping.raster import MappingRaster, MappingType

DATA_DIR = Path(__file__).parent.resolve() / "data"

network_pyproj.set_network_enabled(True)  # type: ignore


@pytest.fixture
def raster():
    # First test wrong datafiles:
    # Test wrong path
    with pytest.raises(FileNotFoundError):
        MappingRaster(DATA_DIR / "random_jpg_wrong.JPG")

    # Test non supported file by rasterio
    with pytest.raises(MappingError):
        MappingRaster(__file__)

    # TODO test wrong CRS. Not sure how to do as the Mapping Raster awaits a CRS class

    # Test CRS system without z Axis
    with pytest.raises(CRSnoZaxisError):
        MappingRaster(DATA_DIR / "dhm_at_lamb_10m_2018.tif")

    # Test error if proload full and window are sprecified
    with pytest.raises(ValueError):
        MappingRaster(
            DATA_DIR / "dhm_at_lamb_10m_2018.tif",
            crs=CRS("EPSG:31287+5778"),
            preload_full_raster=True,
            preload_window=(556530, 429150, 556550, 429200),
        )

    # Test error window sprecified is not the correct length
    with pytest.raises(ValueError):
        MappingRaster(
            DATA_DIR / "dhm_at_lamb_10m_2018.tif",
            crs=CRS("EPSG:31287+5778"),
            preload_window=(429150, 556550, 429200),  # type: ignore
        )

    # Test error crs and force_no_crs are used together
    with pytest.raises(ValueError):
        MappingRaster(
            DATA_DIR / "dhm_at_lamb_10m_2018.tif",
            crs=CRS("EPSG:31287+5778"),
            force_no_crs=True,
        )

    # Test wrong index band
    with pytest.raises(ValueError):
        MappingRaster(DATA_DIR / "dhm_at_lamb_10m_2018.tif", index_band=2, force_no_crs=True)

    # Test file has more than one band no band index was specified
    with pytest.raises(ValueError):
        MappingRaster(DATA_DIR / "random_jpg.JPG", force_no_crs=True)

    # Test wrong geo-transform of raster
    with pytest.raises(MappingError):
        MappingRaster(DATA_DIR / "random_jpg.JPG", index_band=1)

    mapping_raster = MappingRaster(DATA_DIR / "dhm_at_lamb_10m_2018.tif", crs=CRS("EPSG:31287+5778"))

    assert int(mapping_raster.type)

    return mapping_raster


def test_type(raster):
    assert raster.type == MappingType.Raster


def test_from_dict():

    with pytest.raises(KeyError):
        MappingRaster.from_dict(
            {
                "raster_filepath_wrong": (DATA_DIR / "dhm_at_lamb_10m_2018.tif").as_posix(),
                "crs": CRS("EPSG:31287+5778").to_wkt(),
            }
        )

    MappingRaster.from_dict(
        {
            "type": "Raster",
            "raster_filepath": (DATA_DIR / "dhm_at_lamb_10m_2018.tif").as_posix(),
            "crs": CRS("EPSG:31287+5778").to_wkt(),
        }
    )


def test_param_dict(raster):
    assert raster.param_dict == {
        "type": "Raster",
        "raster_filepath": (DATA_DIR / "dhm_at_lamb_10m_2018.tif").as_posix(),
        "crs": CRS("EPSG:31287+5778").to_wkt(),
        "band": 1,
    }


def test_resolution(raster):
    assert np.isclose(raster.resolution, 10.004021836097209)


def test_transform(raster):
    assert True


def test_width(raster):
    assert True


def test_height(raster):
    assert True


def test_load_window():

    # Tested with gdal -> gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 556538.9876 429158.5652
    # Report:
    # Location: (469.512998180733P, 430.567498733493L)
    # Band 1:
    # Value: 1284.07670385579

    # Test window limits are wrong
    with pytest.raises(MappingError):
        MappingRaster(
            DATA_DIR / "dhm_at_lamb_10m_2018.tif",
            crs=CRS("EPSG:31287+5778"),
            preload_window=(-556530, 429150, 556550, 429200),
        )

    mapping_raster = MappingRaster(
        DATA_DIR / "dhm_at_lamb_10m_2018.tif",
        crs=CRS("EPSG:31287+5778"),
        preload_window=(556530, 429150, 556550, 429200),
    )

    assert mapping_raster.georef_array is not None
    result = mapping_raster.georef_array.map_heights_from_coordinates(
        np.array([556538.9876, 429158.5652]), crs_s=mapping_raster.crs
    )
    assert result.ok is True
    assert np.allclose(
        result.coordinates,
        np.array([[556538.9876, 429158.5652, 1284.07670386]]),
        atol=1e-7,
        rtol=0,
    )


def test_get_window_georef_array(raster):
    assert True


def test_pixel_to_coordinate(raster):
    assert True


def test_pixel_valid(raster):
    assert True


def test_coordinate_on_raster(raster):
    assert True


def test_coordinate_to_pixel(raster):
    assert True


def test_get_coordinate_height(raster):
    assert True


def test_intersection_ray(raster):
    assert True


def test_map_coordinates_from_rays(raster):
    point_3d = np.array([[557057, 429465.0, 1520]])

    ray_vec = np.array([[0.3, -0.3, -0.80]])

    ray_crs = CRS("EPSG:31287+5778")
    x = raster.map_coordinates_from_rays(ray_vectors_crs_s=ray_vec, ray_start_crs_s=point_3d, crs_s=ray_crs)

    # No we also test the case we use another CRS
    point_3d = np.array([[557057, 429465.0, 1520 - 156.68]])

    ray_vec = np.array([[0.3, -0.3, -0.80]])

    ray_crs = CRS("EPSG:31287+8881")
    x1 = raster.map_coordinates_from_rays(ray_vectors_crs_s=ray_vec, ray_start_crs_s=point_3d, crs_s=ray_crs)

    assert np.allclose(x.coordinates, x1.coordinates + np.array([0, 0, 156.68]), atol=1e-4, rtol=0)

    # test now rays not intersecting because of wrong direction
    point_3d = np.array([[557057, 429465.0, 1520]])

    ray_vec = np.array([[0.3, -0.3, 0.80]])

    x = raster.map_coordinates_from_rays(ray_vectors_crs_s=ray_vec, ray_start_crs_s=point_3d, crs_s=raster.crs)
    assert x.ok is False

    # test now rays not intersecting because of wrong direction
    # but add second ray with intersection
    point_3d = np.array([[557057, 429465.0, 1520], [557057, 429465.0, 1520]])
    ray_vec = np.array([[0.3, -0.3, 0.80], [0.3, -0.3, -0.80]])

    x = raster.map_coordinates_from_rays(ray_vectors_crs_s=ray_vec, ray_start_crs_s=point_3d, crs_s=raster.crs)
    assert x.ok is True
    np.testing.assert_array_equal(x.mask, np.array([False, True]))
    # parallel lines should be in this case every horizontal line
    point_3d = np.array([[557057, 429465.0, 1520]])
    ray_vec = np.array([[1.0, 0.3, 0.0]])

    x = raster.map_coordinates_from_rays(ray_vectors_crs_s=ray_vec, ray_start_crs_s=point_3d, crs_s=raster.crs)
    assert x.ok is False


def test_map_heights_from_coordinates(raster):
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
    result = raster.map_heights_from_coordinates(coo, raster.crs)
    assert np.allclose(result.coordinates, coo, atol=0.01, rtol=0)

    # Different CRS
    coo_utm = np.array(
        [
            [531550.98696, 5287808.89400, 1289.12332],
            [531547.64402920, 5287818.13604243, 1284.07670385579],
        ]
    )
    crs = CRS("EPSG:25833+5778")

    result = raster.map_heights_from_coordinates(coo_utm, crs)
    assert result.ok is True
    assert np.allclose(result.coordinates, coo_utm, atol=0.01, rtol=0)

    # Edge case on the upper limit -> width of raster = 1011

    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 561950.0 429158.5652
    # Location: (1010.40023729484P,430.567498733493L)
    # Value: 1024.56193042452
    # coordinates in raster crs
    coo = np.array([561950.0, 429158.5652])
    coo_map_heights = raster.map_heights_from_coordinates(coo, raster.crs)
    assert coo_map_heights.ok is True
    assert np.allclose(coo_map_heights.coordinates[0, 2], 1024.56193042452, atol=1e-6, rtol=0)

    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 561955.0 429158.5652
    #   Location: (1010.90003954914P,430.567498733493L)
    #   Value: 1024.73956631826
    coo = np.array([561955.0, 429158.5652])
    coo_map_heights = raster.map_heights_from_coordinates(coo, raster.crs)
    assert coo_map_heights.ok is True
    assert np.allclose(coo_map_heights.coordinates[0, 2], 1024.73956631826, atol=1e-6, rtol=0)

    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 561956.0 429158.5652
    # Location: (1011P,430.567498733493L)
    # Location is off this file! No further details to report.
    coo = np.array([561956.0, 429158.5652])

    coo_map_heights = raster.map_heights_from_coordinates(coo, raster.crs)
    assert coo_map_heights.ok is False

    # Edge case on the lower limit ->
    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 551842.0 429158.5652
    #   Location: (0P,430.567498733493L)
    #    Value: 952.09822577356
    # coordinates in raster crs
    coo = np.array([551842.0, 429158.5652])
    coo_map_heights = raster.map_heights_from_coordinates(coo, raster.crs)
    assert coo_map_heights.ok is True
    assert np.allclose(coo_map_heights.coordinates[0, 2], 952.09822577356, atol=1e-6, rtol=0)

    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 551845.0 429158.5652
    # Report:
    #   Location: (0.299881352577358P,430.567498733493L)
    #   Band 1:
    #     Value: 952.09822577356
    coo = np.array([551845.0, 429158.5652])
    coo_map_heights = raster.map_heights_from_coordinates(coo, raster.crs)
    assert coo_map_heights.ok is True
    assert np.allclose(coo_map_heights.coordinates[0, 2], 952.09822577356, atol=1e-6, rtol=0)

    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 551847.5 429158.5652
    # Report:
    #   Location: (0.549782479727583P,430.567498733493L)
    #   Band 1:
    #     Value: 952.275286000538
    coo = np.array([551847.5, 429158.5652])
    coo_map_heights = raster.map_heights_from_coordinates(coo, raster.crs)
    assert coo_map_heights.ok is True
    assert np.allclose(coo_map_heights.coordinates[0, 2], 952.275286000538, atol=1e-6, rtol=0)

    # gdallocationinfo -r bilinear -geoloc dhm_at_lamb_10m_2018.tif 551845.0 433464.5652
    #  Location: (0.299881352577358P,0.143421380911605L)
    #  Value: 1102.26696777344
    coo = np.array([551845.0, 433464.5652])
    coo_map_heights = raster.map_heights_from_coordinates(coo, raster.crs)
    assert coo_map_heights.ok is True
    assert np.allclose(coo_map_heights.coordinates[0, 2], 1102.26696777344, atol=1e-6, rtol=0)


def test_map_heights_from_coordinates2(raster):
    coo = np.array([[555837, 430141], [555840, 430141], [555836, 430150], [561555.930, 428812.474]])

    result = raster.map_heights_from_coordinates(coo, raster.crs)

    assert result.ok is True
    assert np.allclose(result.coordinates[:, :2], coo, atol=0.01, rtol=0)

    vienna_heights_crs = CRS("EPSG:31287+8881")
    result_2 = raster.map_heights_from_coordinates(coo, crs_s=vienna_heights_crs)
    assert result.ok is True
    assert np.allclose(
        result.coordinates,
        result_2.coordinates + np.array([0, 0, 156.68]),
        atol=1e-6,
        rtol=0,
    )
