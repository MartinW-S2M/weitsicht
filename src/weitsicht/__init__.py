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

"""Top-level package exports for ``weitsicht``.

This module re-exports the most commonly used classes, functions, and exceptions so that
user code can typically import from ``weitsicht`` directly.
"""

import logging
from importlib import metadata as importlibmetadata
from pathlib import Path

try:  # Python 3.11+
    import tomllib  # stdlib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


from weitsicht import cfg
from weitsicht.camera import CameraType, get_camera_from_dict
from weitsicht.camera.base_perspective import CameraBasePerspective
from weitsicht.camera.opencv_perspective import CameraOpenCVPerspective
from weitsicht.exceptions import (
    CoordinateTransformationError,
    CRSInputError,
    CRSnoZaxisError,
    MapperMissingError,
    MappingBackendError,
    MappingError,
    NotGeoreferencedError,
    WeitsichtError,
)
from weitsicht.geometry.coplanar_collinear import is_coplanar
from weitsicht.geometry.interpolation_bilinear import bilinear_interpolation
from weitsicht.geometry.intersection_bilinear import multilinear_poly_intersection
from weitsicht.geometry.intersection_plane import intersection_plane, intersection_plane_mat_operation
from weitsicht.geometry.line_grid_intersection import line_grid_intersection_points, raster_index_p1_p2, vector_projection
from weitsicht.image import get_image_from_dict
from weitsicht.image.base_class import ImageBase, ImageType
from weitsicht.image.image_batch import ImageBatch
from weitsicht.image.ortho import ImageOrtho
from weitsicht.image.perspective import ImagePerspective
from weitsicht.mapping import get_mapper_from_dict
from weitsicht.mapping.base_class import MappingBase, MappingType
from weitsicht.mapping.georef_array import MappingGeorefArray
from weitsicht.mapping.horizontal_plane import MappingHorizontalPlane
from weitsicht.mapping.map_trimesh import MappingTrimesh
from weitsicht.mapping.raster import MappingRaster
from weitsicht.metadata.camera_estimator_metadata import estimate_camera, ior_from_meta
from weitsicht.metadata.eor_from_meta import eor_from_meta
from weitsicht.metadata.image_from_meta import ImageFromMetaBuilder, image_from_meta
from weitsicht.metadata.tag_systems import PyExifToolTags
from weitsicht.transform.coordinates_transformer import CoordinateTransformer
from weitsicht.transform.rotation import Rotation
from weitsicht.type_guards import (
    is_camera_type,
    is_opencv_camera,
    is_ortho_image,
    is_perspective_image,
)
from weitsicht.utils import (
    Array3x3,
    ArrayN_,
    ArrayNx2,
    ArrayNx3,
    ArrayNxN,
    Issue,
    MappingResult,
    MappingResultSuccess,
    ProjectionResult,
    ProjectionResultSuccess,
    ResultFailure,
    Vector2D,
    Vector3D,
)

__all__ = [
    "CameraType",
    "get_camera_from_dict",
    "CameraOpenCVPerspective",
    "CameraBasePerspective",
    "ImageType",
    "get_image_from_dict",
    "ImagePerspective",
    "ImageBase",
    "ImageOrtho",
    "ImageBatch",
    "MappingType",
    "get_mapper_from_dict",
    "MappingBase",
    "MappingRaster",
    "MappingTrimesh",
    "MappingHorizontalPlane",
    "MappingGeorefArray",
    "Rotation",
    "CoordinateTransformer",
    "PyExifToolTags",
    "ior_from_meta",
    "estimate_camera",
    "image_from_meta",
    "ImageFromMetaBuilder",
    "eor_from_meta",
    "is_opencv_camera",
    "is_camera_type",
    "is_perspective_image",
    "is_ortho_image",
    "ArrayNx3",
    "ArrayNx2",
    "Vector2D",
    "Vector3D",
    "Array3x3",
    "ArrayN_",
    "ArrayNxN",
    "Issue",
    "is_coplanar",
    "bilinear_interpolation",
    "multilinear_poly_intersection",
    "intersection_plane",
    "intersection_plane_mat_operation",
    "vector_projection",
    "raster_index_p1_p2",
    "line_grid_intersection_points",
    "ProjectionResult",
    "MappingResult",
    "MappingResultSuccess",
    "ResultFailure",
    "ProjectionResultSuccess",
    "cfg",
    "allow_ballpark_transformations",
    "allow_non_best_transformations",
    "NotGeoreferencedError",
    "MapperMissingError",
    "MappingError",
    "MappingBackendError",
    "WeitsichtError",
    "CoordinateTransformationError",
    "CRSInputError",
    "CRSnoZaxisError",
]


def _read_pyproject_version() -> str:
    """Read the package version from ``pyproject.toml``.

    This is used as a fallback if the distribution metadata is not available (e.g. running
    from a source checkout without an installed wheel).

    :return: Version string.
    :rtype: str
    """
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.is_file():
        return "0.0.0-dev"
    data = tomllib.loads(pyproject.read_text())
    if "project" in data and "version" in data["project"]:
        return data["project"]["version"]
    if "tool" in data and "poetry" in data["tool"] and "version" in data["tool"]["poetry"]:
        return data["tool"]["poetry"]["version"]
    return "0.0.0-dev"


try:
    __version__ = importlibmetadata.version("weitsicht")
except importlibmetadata.PackageNotFoundError:
    __version__ = _read_pyproject_version()
__author__ = "Martin Wieser"

logger = logging.getLogger(__name__)


def allow_ballpark_transformations(allow: bool = True) -> None:
    """Allow or disallow "ballpark" coordinate transformations.

    This toggles ``pyproj`` transformations that may use less-accurate but available
    fallbacks. Defaults to ``True``.

    :param allow: Whether to allow ballpark transformations, defaults to ``True``.
    :type allow: bool
    """
    cfg._ballpark_transformation = allow


def allow_non_best_transformations(allow: bool = True) -> None:
    """Allow or disallow non-best coordinate transformations.

    If set to ``False`` (default behavior in weitsicht), only "best available" transformations
    are used. Defaults to ``True``.

    :param allow: Whether to allow non-best transformations, defaults to ``True``.
    :type allow: bool
    """
    cfg._only_best_transformation = not allow
