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

"""Geometry helper functions used across weitsicht."""

import logging

from weitsicht.geometry.coplanar_collinear import is_coplanar
from weitsicht.geometry.interpolation_bilinear import bilinear_interpolation
from weitsicht.geometry.intersection_bilinear import multilinear_poly_intersection
from weitsicht.geometry.intersection_plane import intersection_plane, intersection_plane_mat_operation
from weitsicht.geometry.line_grid_intersection import line_grid_intersection_points, raster_index_p1_p2, vector_projection

__all__ = [
    "intersection_plane",
    "intersection_plane_mat_operation",
    "vector_projection",
    "raster_index_p1_p2",
    "line_grid_intersection_points",
    "multilinear_poly_intersection",
    "bilinear_interpolation",
    "bilinear_interpolation",
    "is_coplanar",
]

logger = logging.getLogger(__name__)
