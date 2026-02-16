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

"""Factory helpers for creating camera models from dictionaries."""

from __future__ import annotations

from weitsicht.camera.base_perspective import CameraBasePerspective
from weitsicht.camera.camera_types import CameraType
from weitsicht.camera.opencv_perspective import CameraOpenCVPerspective

__all__ = ["get_camera_from_dict"]


def get_camera_from_dict(camera_dict: dict) -> CameraBasePerspective:
    """Create a camera model from a parameter dictionary.

    Exceptions raised by the underlying camera ``from_dict`` implementations are propagated unchanged.

    :param camera_dict: Dictionary as received by :attr:`CameraBasePerspective.param_dict`.
    :type camera_dict: dict
    :return: Camera model instance.
    :rtype: CameraBasePerspective
    :raises KeyError: If the dictionary key ``type`` is missing or required camera parameters are missing.
    :raises ValueError: If configuration values are invalid or the camera type is unsupported.
    :raises TypeError: If configuration values have incompatible types.
    """

    camera_type = camera_dict.get("type", None)
    if camera_type is None:
        raise KeyError("Dictionary key 'type' is missing")

    if camera_dict.get("type") == CameraType.OpenCV.fullname:
        return CameraOpenCVPerspective.from_dict(camera_dict)

    raise ValueError(f"Unsupported camera type {camera_type!r}")
