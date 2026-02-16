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

import weitsicht
from weitsicht.exceptions import CoordinateTransformationError
from weitsicht.transform.coordinates_old import CoordinatesTransformer


def test_from_crs():
    source_crs = CRS(4979)
    target_crs = CRS(25833)
    # Test for Array 1x3
    points = np.array([16, 48, 100])
    transformer = CoordinatesTransformer.from_crs(crs_s=source_crs, crs_t=target_crs, points=points)

    assert transformer


def test_from_crs_fails():

    source_crs = CRS(4979)
    points = np.array([48, 400, 0])

    target_crs = CRS(25833)

    with pytest.raises(CoordinateTransformationError):
        CoordinatesTransformer.from_crs(crs_s=source_crs, crs_t=target_crs, points=points)


def test_is_point():
    source_crs = CRS(4979)

    points = np.array([16, 48])
    transformer = CoordinatesTransformer(crs=source_crs, points=points)
    assert transformer.is_point

    # Test for 3d Point
    points = np.array([16, 48, 100])
    transformer = CoordinatesTransformer(crs=source_crs, points=points)
    assert transformer.is_point

    points = np.array([[16, 48], [16.1, 48]])
    transformer = CoordinatesTransformer(crs=source_crs, points=points)
    assert not transformer.is_point


def test_is_3d():

    source_crs = CRS(4979)

    # Test for Array 1x3
    points = np.array([16, 48, 100])
    transformer = CoordinatesTransformer(crs=source_crs, points=points)
    assert transformer.is_3d

    # Test for Array 2x3
    points = np.array([[16, 48, 100], [16, 48, 100]])
    transformer = CoordinatesTransformer(crs=source_crs, points=points)
    assert transformer.is_3d

    # Test that 2d is really 2d
    points = np.array([16, 48])
    transformer = CoordinatesTransformer(crs=source_crs, points=points)
    assert not transformer.is_3d


def test_geojson():
    source_crs = CRS(4979)
    points = np.array([48, 16, 0])

    transformer = CoordinatesTransformer(crs=source_crs, points=points)

    assert transformer.geojson("Point") == {
        "type": "Point",
        "coordinates": (48.0, 16.0, 0.0),
    }

    line = np.array([[16, 48, 100], [16, 48, 100]])
    transformer = CoordinatesTransformer(crs=source_crs, points=line)

    assert transformer.geojson("LineString") == {
        "type": "LineString",
        "coordinates": ((16.0, 48.0, 100.0), (16.0, 48.0, 100.0)),
    }


def test_geojson_fails():
    source_crs = CRS(4979)
    line = np.array([[16, 48, 100], [16, 48, 100]])

    transformer = CoordinatesTransformer(crs=source_crs, points=line)

    with pytest.raises(CoordinateTransformationError):
        transformer.geojson("Point")

    with pytest.raises(CoordinateTransformationError):
        transformer.geojson("Polygon")


def test_to_crs():

    source_crs = CRS(4979)
    points = np.array([16, 48, 0])

    target_crs = CRS("epsg:31286+5730")
    weitsicht.allow_non_best_transformations()
    weitsicht.allow_ballpark_transformations()

    transformer = CoordinatesTransformer(crs=source_crs, points=points)

    result = transformer.to_crs(crs_target=target_crs)

    assert result


def test_to_crs_fails():

    source_crs = CRS(4979)
    points = np.array([48, 400, 0])

    target_crs = CRS(25833)

    transformer = CoordinatesTransformer(crs=source_crs, points=points)

    with pytest.raises(CoordinateTransformationError):
        transformer.to_crs(crs_target=target_crs)
