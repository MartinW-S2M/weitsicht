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
"""Shared utility types, conversion helpers, and result objects.

This module defines:
- Numpy typing aliases used throughout the package (``ArrayNx2``, ``ArrayNx3``, …)
- Helper functions for normalizing array-like inputs (``to_array_nx2``, ``to_array_nx3``)
- Result dataclasses used by projection and mapping APIs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Literal, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

# from typing_extensions import Annotated
from pyproj import CRS
from pyproj.crs.crs import CompoundCRS

__all__ = [
    "MappingResultSuccess",
    "ProjectionResult",
    "ProjectionResultSuccess",
    "MappingResult",
    "MappingResultSuccess",
    "ResultFailure",
    "Issue",
    "to_array_nx2",
    "to_array_nx3",
    "MaskN_",
    "ArrayN_",
    "Array3x3",
    "ArrayNx3",
    "ArrayNx2",
    "ArrayNxN",
    "Vector2D",
    "Vector3D",
]

_N = TypeVar("_N", bound=int)
_IssueEnum = TypeVar("_IssueEnum", bound=Enum)
#
## MaskN_: TypeAlias = npt.NDArray[np.bool_]
MaskN_: TypeAlias = NDArray[np.bool_]
ArrayN_: TypeAlias = NDArray[np.float64]

Array3x3: TypeAlias = NDArray  # [tuple[Literal[3], Literal[3]], np.dtype[np.float64]]
ArrayNx3: TypeAlias = NDArray  #  [tuple[_N, Literal[3]], np.dtype[np.float64]]
ArrayNx2: TypeAlias = NDArray  #  [tuple[_N, Literal[2]], np.dtype[np.float64]]
ArrayNxN: TypeAlias = NDArray  #  [tuple[_N, _N], np.dtype[np.float64]]

Vector2D: TypeAlias = NDArray  #  [tuple[Literal[2]], np.dtype[np.float64]]
Vector3D: TypeAlias = NDArray  #  [tuple[Literal[3]], np.dtype[np.float64]]

# _N = TypeVar("_N", bound=int)

# Typed ndarray aliases with shape hints for stricter type checking
# MaskN_: TypeAlias = NDArray[np.bool_]  # shape: (*,)
# ArrayN_: TypeAlias = NDArray[np.float64]  # shape: (*,)

# Shape annotations via Annotated to avoid numpy.typing.Shape dependency
# Array3x3: TypeAlias = Annotated[NDArray[np.float64], "...x3x3"]
# ArrayNx3: TypeAlias = Annotated[NDArray[np.float64], "...xN x3"]
# ArrayNx2: TypeAlias = Annotated[NDArray[np.float64], "...xN x2"]
# ArrayNxN: TypeAlias = NDArray[np.float64]

# Vector2D: TypeAlias = Annotated[NDArray[np.float64], "len=2"]
# Vector3D: TypeAlias = Annotated[NDArray[np.float64], "len=3"]


def to_array_nx2(
    array_like_nx2: list[int] | list[list[int]] | list[float] | list[list[float]] | ArrayNx2 | ArrayNx3,
) -> ArrayNx2:
    """Convert array-like input to an N×2 numpy array.

    :param array_like_nx2: Array-like input interpreted as 2D coordinates.
    :type array_like_nx2: list[int] | list[list[int]] | list[float] | list[list[float]] | ArrayNx2 | ArrayNx3
    :return: Array of shape N×2.
    :rtype: ArrayNx2
    :raises ValueError: If the input cannot be interpreted as an N×2 array.
    """

    _array_like_nx2 = np.array(array_like_nx2)
    if _array_like_nx2.ndim == 1:
        _array_like_nx2 = np.array([_array_like_nx2])

    if _array_like_nx2.shape[1] >= 2:
        _array_like_nx2 = _array_like_nx2[:, :2]
    else:
        raise ValueError("Dimensions of 2D array not fitting")

    return _array_like_nx2


def to_array_nx3(
    array_like_nx3: list[float] | list[list[float]] | ArrayNx2 | ArrayNx3, fill_z: float | None = None
) -> ArrayNx3:
    """Convert array-like input to an N×3 numpy array.

    If the input is N×2 and ``fill_z`` is provided, a Z column is appended. Otherwise a
    :class:`ValueError` is raised for inputs that cannot be interpreted as N×3.

    :param array_like_nx3: Array-like input interpreted as 3D coordinates.
    :type array_like_nx3: list[float] | list[list[float]] | ArrayNx2 | ArrayNx3
    :param fill_z: Z value used when the input is N×2, defaults to ``None``.
    :type fill_z: float | None
    :return: Array of shape N×3.
    :rtype: ArrayNx3
    :raises ValueError: If the input cannot be interpreted as an N×3 array.
    """

    _array_like_nx3 = np.array(array_like_nx3)

    if _array_like_nx3.ndim == 1:
        _array_like_nx3 = np.array([_array_like_nx3])

    if _array_like_nx3.shape[1] >= 3:
        _array_like_nx3 = _array_like_nx3[:, :3]
    else:
        if fill_z is not None:
            _array_like_nx3 = np.hstack((_array_like_nx3, np.zeros((_array_like_nx3.shape[0], 1))))
        else:
            raise ValueError("Dimension of 3D array not fitting")

    return _array_like_nx3


class Issue(Enum):
    """Issue codes used in result objects to describe runtime outcomes."""

    # POSITION_MISSING = "position_missing"
    # ORIENTATION_MISSING = "orientation_missing"
    # CAMERA_MISSING = "camera_missing"
    # CRS_MISSING = "crs_missing"
    # MAPPER_MISSING = "mapper_missing"
    # GEO_TRANSFORM_MISSING = "Geo Transform for orthophoto missing"
    # DIMENSION_MISMATCH = "Dimensions do not match"
    WRONG_DIRECTION = "ray probably in the wrong direction"
    OUTSIDE_RASTER = "ray is outside raster"
    RASTER_NO_DATA = "touched no data raster cells"
    MAX_ITTERATION = "max itteration reached"
    NO_INTERSECTION = "No intersection was found"
    INVALID_PROJECTIIONS = "Projections are not within image border"
    IMAGE_BATCH_ERROR = "A single image in image batch calls raised an error"
    UNKNOWN = "unknown"


@dataclass
class ResultFailure(Generic[_IssueEnum]):
    """Failure result for projection and mapping operations."""

    ok: Literal[False]
    error: str
    issues: set[_IssueEnum] = field(default_factory=set)


@dataclass
class ProjectionResultSuccess:
    """Successful projection result."""

    ok: Literal[True]
    pixels: ArrayNx2
    mask: MaskN_
    issues: set[Issue] = field(default_factory=lambda: set[Issue]())


ProjectionResult = ProjectionResultSuccess | ResultFailure[Issue]


@dataclass
class MappingResultSuccess:
    """Successful mapping result."""

    ok: Literal[True]
    coordinates: ArrayNx3
    mask: MaskN_  # mapped 3D coords
    crs: CRS | CompoundCRS | None = None
    gsd: float | None = None  # mean GSD (or scalar for center)
    gsd_per_point: ArrayN_ | None = None  # optional per-point GSDs
    area: float | None = None
    issues: set[Issue] = field(default_factory=lambda: set[Issue]())


# @dataclass
# class MappingResult:
#    ok: bool
#    coordinates: Optional[np.ndarray] = None
#    crs: Optional[CRS | CompoundCRS | None] = None
#    mask: Optional[np.ndarray] = None   # mapped 3D coords
#    gsd: Optional[float] = None           # mean GSD (or scalar for center)
#    gsd_per_point: Optional[np.ndarray] = None  # optional per-point GSDs
#    area: Optional[float] = None          # footprint area (if applicable)
#    error: Optional[list[str]] = None
#    issues: set[Issue] = field(default_factory=set)   # e.g. ["position", "orientation"]


MappingResult = MappingResultSuccess | ResultFailure[Issue]
