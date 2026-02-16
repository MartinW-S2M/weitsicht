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

"""Factory helpers for creating image models from dictionaries."""

from weitsicht.image.base_class import ImageBase, ImageType
from weitsicht.image.ortho import ImageOrtho
from weitsicht.image.perspective import ImagePerspective
from weitsicht.mapping.base_class import MappingBase

__all__ = ["get_image_from_dict"]


def get_image_from_dict(param_dict: dict, mapper: MappingBase | None = None) -> ImageBase | ImageOrtho | ImagePerspective:
    """Create an image model from a parameter dictionary.

    Exceptions raised by the underlying image ``from_dict`` implementations are propagated unchanged.

    :param param_dict: Dictionary as received by :attr:`~weitsicht.image.base_class.ImageBase.param_dict`.
    :type param_dict: dict
    :param mapper: Mapping instance, defaults to ``None``.
    :type mapper: MappingBase | None
    :return: Image model instance.
    :rtype: ImageBase
    :raises KeyError: If the dictionary key ``type`` is missing or required image parameters are missing.
    :raises ValueError: If configuration values are invalid or the image type is unsupported.
    :raises TypeError: If configuration values have incompatible types.
    :raises CRSInputError: If a CRS WKT string is invalid or unsupported.
    """
    image_type = param_dict.get("type", None)
    if image_type is None:
        raise KeyError("Dictionary key 'type' is missing")

    if image_type == ImageType.Orthophoto.fullname:
        return ImageOrtho.from_dict(param_dict, mapper)

    elif image_type == ImageType.Perspective.fullname:
        return ImagePerspective.from_dict(param_dict, mapper)

    raise ValueError(f"Unsupported image type {image_type!r}")
