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
"""Mapping backends and factory helpers.

The mapping package contains mapper implementations (raster, mesh, plane, …) and utilities
for constructing mappers from configuration dictionaries.
"""

import logging

from weitsicht.mapping.base_class import MappingBase, MappingType
from weitsicht.mapping.georef_array import MappingGeorefArray
from weitsicht.mapping.horizontal_plane import MappingHorizontalPlane
from weitsicht.mapping.map_trimesh import MappingTrimesh
from weitsicht.mapping.mapping_dict_selector import get_mapper_from_dict
from weitsicht.mapping.raster import MappingRaster

__all__ = [
    "MappingBase",
    "MappingType",
    "MappingGeorefArray",
    "MappingHorizontalPlane",
    "MappingTrimesh",
    "MappingRaster",
    "get_mapper_from_dict",
]

logger = logging.getLogger(__name__)
