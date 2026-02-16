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

"""Type guard helpers for narrowing weitsicht base classes.

These helpers are primarily intended for static type checkers (mypy, pyright).
They allow narrowing from ``ImageBase``/``CameraBasePerspective`` to concrete subclasses
based on the runtime type discriminator enums.
"""

from __future__ import annotations

from typing import Literal, TypeGuard, overload

from weitsicht.camera.base_perspective import CameraBasePerspective, CameraType
from weitsicht.camera.opencv_perspective import CameraOpenCVPerspective
from weitsicht.image.base_class import ImageBase, ImageType
from weitsicht.image.ortho import ImageOrtho
from weitsicht.image.perspective import ImagePerspective

__all__ = [
    "is_opencv_camera",
    "is_camera_type",
    "is_perspective_image",
    "is_ortho_image",
]


# Camera guards
def is_opencv_camera(cam: CameraBasePerspective) -> TypeGuard[CameraOpenCVPerspective]:
    """Return whether ``cam`` is an OpenCV camera.

    :param cam: Camera instance to check.
    :type cam: CameraBasePerspective
    :return: ``True`` if ``cam`` is an OpenCV-based camera.
    :rtype: TypeGuard[CameraOpenCVPerspective]
    """
    return cam.type is CameraType.OpenCV


# Overload-like behavior via Literal plus a fallback; extend when new camera types are added.
@overload
def is_camera_type(
    cam: CameraBasePerspective, kind: Literal[CameraType.OpenCV]
) -> TypeGuard[CameraOpenCVPerspective]: ...


@overload
def is_camera_type(cam: CameraBasePerspective, kind: CameraType) -> bool: ...


def is_camera_type(cam: CameraBasePerspective, kind: CameraType) -> bool:
    """Return whether ``cam.type`` matches ``kind``.

    When called with a literal discriminator, type checkers can narrow the return type
    accordingly (e.g. ``CameraType.OpenCV`` → :class:`CameraOpenCVPerspective`).

    :param cam: Camera instance to check.
    :type cam: CameraBasePerspective
    :param kind: Expected camera type discriminator.
    :type kind: CameraType
    :return: ``True`` if ``cam.type`` matches ``kind``.
    :rtype: bool
    """
    return cam.type is kind


# Image guards
def is_perspective_image(img: ImageBase) -> TypeGuard[ImagePerspective]:
    """Return whether ``img`` is a perspective image.

    :param img: Image instance to check.
    :type img: ImageBase
    :return: ``True`` if ``img`` is a perspective image.
    :rtype: TypeGuard[ImagePerspective]
    """
    return img.type is ImageType.Perspective


def is_ortho_image(img: ImageBase) -> TypeGuard[ImageOrtho]:
    """Return whether ``img`` is an orthophoto image.

    :param img: Image instance to check.
    :type img: ImageBase
    :return: ``True`` if ``img`` is an orthophoto image.
    :rtype: TypeGuard[ImageOrtho]
    """
    return img.type is ImageType.Orthophoto
