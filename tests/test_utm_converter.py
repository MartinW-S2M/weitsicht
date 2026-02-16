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
from weitsicht.transform.utm_converter import get_zone, point_convert_utm_wgs84_egm2008


def test_get_zone():
    utm_zone = get_zone(longitude=16, latitude=48)
    assert utm_zone == 32633


def test_get_zone_fails():
    with pytest.raises(ValueError):
        get_zone(longitude=200, latitude=48)


def test_wgs84_utm_point_converter():
    weitsicht.allow_ballpark_transformations()
    weitsicht.allow_non_best_transformations()
    x_utm, y_utm, z_geoid, utm_crs = point_convert_utm_wgs84_egm2008(CRS(4979), x=16, y=18, z=100)

    # We test only for x and y, Z is configured to use geoid data which depending on your interpreters
    # pyproj settings could have the geoid file
    assert np.linalg.norm(np.array([x_utm, y_utm]) - np.array([(605866.998, 1990471.052)])) < 0.001
    assert utm_crs.to_wkt() == CRS("EPSG:32633+3855").to_wkt()

    x_utm, y_utm, z_geoid, utm_crs = point_convert_utm_wgs84_egm2008(
        CRS("EPSG:32633"), x=605866.998, y=1990471.052, z=100
    )
