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

"""Camera type definitions used across the weitsicht camera module."""

from enum import Enum

## Moved out of base_perspective if other cameras are not of type BasePerspective

__all__ = ["CameraType"]


class CameraType(Enum):
    """Enum of supported camera model types.

    The ``fullname`` attribute is used for serialization (e.g. in ``param_dict``)
    and must match what factory functions expect (e.g. ``get_camera_from_dict``).
    Add new camera model types here.
    """

    fullname: str
    Base = 1, "Base"
    OpenCV = 2, "OpenCV"

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value
