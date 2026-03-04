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

"""Coordinate transformations and rotation helpers."""

import logging

from weitsicht.transform.coordinates_transformer import CoordinateTransformer
from weitsicht.transform.rotation import Rotation
from weitsicht.transform.utm_converter import get_zone, point_convert_utm_wgs84_egm2008, point_wgs84ell_to_utm
from weitsicht.transform.wgs84_local_tangent import WGS84LocalTangent

__all__ = [
    "CoordinateTransformer",
    "Rotation",
    "WGS84LocalTangent",
    "get_zone",
    "point_convert_utm_wgs84_egm2008",
    "point_wgs84ell_to_utm",
]

logger = logging.getLogger(__name__)
