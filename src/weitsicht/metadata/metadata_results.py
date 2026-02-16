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


"""Result objects and issues for metadata extraction and image building.

This mirrors the pattern used for projection and mapping results, but keeps metadata issues
separate from the generic :class:`weitsicht.utils.Issue` enum.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from pyproj import CRS
from pyproj.crs.crs import CompoundCRS

from weitsicht.camera.base_perspective import CameraBasePerspective
from weitsicht.image.perspective import ImagePerspective
from weitsicht.transform.rotation import Rotation
from weitsicht.utils import ResultFailure

__all__ = [
    "MetadataIssue",
    "IORFromMetaResultSuccess",
    "IORFromMetaResult",
    "EORFromMetaResultSuccess",
    "EORFromMetaResult",
    "ImageFromMetaResultSuccess",
    "ImageFromMetaResult",
]


class MetadataIssue(Enum):
    """Issue codes for metadata extraction and estimation."""

    IOR_FAILED = "Metadata IOR extraction failed"
    EOR_FAILED = "Metadata EOR extraction failed"
    MISSING_GPS = "Metadata GPS tags missing"
    MISSING_ORIENTATION = "Metadata orientation tags missing"
    TRANSFORMATION_FAILED = "Metadata coordinate transformation failed"
    UNKNOWN = "unknown"


@dataclass
class IORFromMetaResultSuccess:
    """Successful IOR (intrinsics) estimation from metadata."""

    ok: Literal[True]
    camera: CameraBasePerspective
    width: int
    height: int


IORFromMetaResult = IORFromMetaResultSuccess | ResultFailure[MetadataIssue]


@dataclass
class EORFromMetaResultSuccess:
    """Successful EOR (pose) extraction from metadata."""

    ok: Literal[True]
    position: np.ndarray
    orientation: Rotation
    crs: CRS | CompoundCRS | None


EORFromMetaResult = EORFromMetaResultSuccess | ResultFailure[MetadataIssue]


@dataclass
class ImageFromMetaResultSuccess:
    """Successful image build from metadata."""

    ok: Literal[True]
    image: ImagePerspective
    ior: IORFromMetaResultSuccess
    eor: EORFromMetaResultSuccess


ImageFromMetaResult = ImageFromMetaResultSuccess | ResultFailure[MetadataIssue]
