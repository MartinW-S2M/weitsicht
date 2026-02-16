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


"""Exceptions used in the package.

Design goal:
- Provide a small, well-defined hierarchy so users can either catch one base class
or handle specific subclasses.
"""


class WeitsichtError(Exception):
    """Base exception for the weitsicht package."""


class NotGeoreferencedError(WeitsichtError):
    """No valid georeference is provided for images"""


class CRSInputError(WeitsichtError, ValueError):
    """Invalid CRS/transformer input was provided."""


class CRSnoZaxisError(CRSInputError):
    """CRS does not define a vertical (Z) axis."""


class CoordinateTransformationError(WeitsichtError):
    """Coordinate Conversion is not possible"""


class MappingError(WeitsichtError):
    """Mapping of the geometry was not possible"""


class MappingBackendError(MappingError):
    """A mapping backend raised an unexpected/foreign exception."""


class MapperMissingError(MappingError):
    """No valid mapper is provided for image mapping functions"""
