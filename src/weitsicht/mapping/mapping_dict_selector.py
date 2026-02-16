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

"""Factory helpers for creating mapping backends from configuration dictionaries."""

from __future__ import annotations

from weitsicht.mapping.base_class import MappingBase, MappingType
from weitsicht.mapping.horizontal_plane import MappingHorizontalPlane
from weitsicht.mapping.map_trimesh import MappingTrimesh
from weitsicht.mapping.raster import MappingRaster

__all__ = ["get_mapper_from_dict"]


def get_mapper_from_dict(
    mapper_config: dict,
) -> MappingBase:
    """Load a mapper instance from a configuration dictionary.

    The configuration must include a ``type`` key matching an implemented mapper type
    (e.g. ``horizontalPlane``, ``Raster``, ``Trimesh``).

    Exceptions raised by the underlying mapper ``from_dict`` implementations are propagated unchanged.

    :param mapper_config: Mapper configuration dictionary (typically created via ``mapper.param_dict``).
    :type mapper_config: dict
    :return: Instantiated mapper.
    :rtype: MappingBase
    :raises KeyError: If the dictionary key ``type`` is missing.
    :raises FileNotFoundError: If a referenced raster/mesh file does not exist.
    :raises ValueError: If configuration values are invalid or the mapper type is unsupported.
    :raises CRSInputError: If CRS input in the configuration is invalid (e.g. malformed WKT).
    :raises CRSnoZaxisError: If a required CRS does not define a Z axis.
    :raises MappingError: If the mapper cannot be initialized.
    """

    mapper_type = mapper_config.get("type", None)
    if mapper_type is None:
        raise KeyError("Dictionary key 'type' is missing")

    if mapper_type == MappingType.HorizontalPlane.fullname:
        mapper = MappingHorizontalPlane.from_dict(mapper_config)

    elif mapper_type == MappingType.Raster.fullname:
        mapper = MappingRaster.from_dict(mapper_config)

    elif mapper_type == MappingType.Trimesh.fullname:
        mapper = MappingTrimesh.from_dict(mapper_config)

    else:
        raise ValueError(f"Unsupported mapper type {mapper_type!r}")

    return mapper
