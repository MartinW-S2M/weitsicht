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

"""Batch operations for mapping and projecting across multiple images."""

import logging
from typing import overload

import numpy as np
from pyproj import CRS

from weitsicht.exceptions import (
    CoordinateTransformationError,
    CRSInputError,
    MapperMissingError,
    MappingError,
    NotGeoreferencedError,
)
from weitsicht.image.base_class import ImageBase
from weitsicht.mapping.base_class import MappingBase
from weitsicht.transform.coordinates_transformer import CoordinateTransformer
from weitsicht.utils import ArrayNx3, Issue, MappingResult, ProjectionResult, ResultFailure, to_array_nx3

__all__ = ["ImageBatch"]

logger = logging.getLogger(__name__)


class ImageBatch:
    """Container for running operations on a set of named images."""

    def __init__(self, images: dict[str, ImageBase], mapper: MappingBase | None = None):
        """Initialize an image batch.

        :param images: Dictionary of images.
        :type images: dict[str, ImageBase]
        :param mapper: Optional default mapper assigned to images without a mapper, defaults to ``None``.
        :type mapper: MappingBase | None
        """
        self.images: dict[str, ImageBase] = images
        self.mapper: MappingBase | None = mapper

        if self.mapper is not None:
            for name, img in self.images.items():
                if img.mapper is None:
                    self.images[name].mapper = self.mapper

    def __len__(self):
        """Return the number of images in the batch."""

        return len(self.images)

    def add_images(self, images: dict[str, ImageBase]):
        """Add images to the batch.

        :param images: Images to add.
        :type images: dict[str, ImageBase]
        :raises KeyError: If any keys already exist in the batch.
        """

        equal_keys = list(set(self.images.keys()).intersection(images.keys()))

        if len(equal_keys) > 0:
            raise KeyError("Identical keys can not be added")

        if self.mapper is not None:
            for name, img in images.items():
                if img.mapper is None:
                    images[name].mapper = self.mapper

        self.images.update(images)

    def __setitem__(self, key, image: ImageBase):
        """Set an image in the batch by key.

        :param key: Image key.
        :param image: Image instance.
        :type image: ImageBase
        :raises ValueError: If the object is not an image instance.
        """

        if image.__class__.__base__ != ImageBase:
            raise ValueError("Only a image class can be assigned")

        self.images[key] = image

    @overload
    def __getitem__(self, keys: str) -> ImageBase: ...

    @overload
    def __getitem__(self, keys: tuple[str] | list[str]) -> dict[str, ImageBase]: ...

    def __getitem__(self, keys: str | tuple[str] | list[str]) -> ImageBase | dict[str, ImageBase]:
        """Get images by key or keys.

        :param keys: Image key or sequence of keys.
        :type keys: str | tuple[str] | list[str]
        :return: Single image or a dictionary of images.
        :rtype: ImageBase | dict[str, ImageBase]
        :raises KeyError: If a requested key is not found.
        :raises TypeError: If ``keys`` has an unsupported type.
        """

        if isinstance(keys, str):
            if keys in self.images.keys():
                return self.images[keys]
            else:
                raise KeyError("Key not found in images")
        elif isinstance(keys, tuple) or isinstance(keys, list):
            # raise KeyError("Key not found in images")
            # convert a simple index and tuples to lists
            # if isinstance(keys, tuple)
            # keys = list(keys) if type(keys) in [list, tuple] else list([keys])

            dict_return = {}

            for key in keys:
                if key in self.images.keys():
                    dict_return[key] = self.images[key]
                else:
                    raise KeyError("Keys not found in images")

            if len(dict_return) == 1:
                return list(dict_return.values())[0]
            return dict_return

        raise TypeError

    @overload
    def index(self, indices: int) -> ImageBase: ...

    @overload
    def index(self, indices: list[int] | tuple[int]) -> dict[str, ImageBase]: ...
    def index(self, indices: list[int] | tuple[int] | int) -> ImageBase | dict[str, ImageBase]:
        """index get images by indices

        Get the images by using indices. Will return the image at the indices of the image dict from ImageBatch.

        :param indices: The indices you want to get images for
        :type indices: list[int] | tuple[int] | int
        :raises TypeError: Indices type not valid
        :raises IndexError: One of the indices is not valid
        :return: return single image or dict of images for multiple indices
        :rtype: ImageBase | dict[str, ImageBase]
        """

        if isinstance(indices, int):
            _indices = list([indices])

        elif isinstance(indices, tuple) or isinstance(indices, list):
            _indices = list(indices)
        else:
            raise TypeError

        keys = list(self.images.keys())

        try:
            dict_return = {keys[x]: self.images[keys[x]] for x in _indices}

        except IndexError as err:
            raise IndexError("Index not found in images") from err

        if len(dict_return) == 1:
            return list(dict_return.values())[0]

        return dict_return

    def map_center_point(
        self, mapper: MappingBase | None = None, transformer: CoordinateTransformer | None = None
    ) -> dict[str, MappingResult]:
        """Map center point of image to 3d

        :param mapper: Specify Mapper to be used, can be different to the one assigned in the class
        :param transformer: A CoordinateTransfoer object which can be passed instead using crs_s
        :return: MappingResult for all images
        :raises ValueError: Image Batch is empty
        :raises NotGeoreferencedError: If an image is not geo-referenced
        :raises MapperMissingError: If no mapper is available
        :raises CRSInputError: If CRS/transformer input is invalid
        :raises CoordinateTransformationError: If coordinate transformation fails
        :raises MappingError: If mapping fails
        """

        center_p_dict = {}

        if self.__len__() == 0:
            raise ValueError("ImageBatch is empty")

        # one_mapping_worked = False
        all_mapping_worked = True
        for key, image in self.images.items():
            try:
                result = image.map_center_point(mapper=self.mapper if mapper is None else mapper, transformer=transformer)

            except NotGeoreferencedError as err_georef:
                result = ResultFailure(ok=False, error=str(err_georef), issues={Issue.IMAGE_BATCH_ERROR})
            except MapperMissingError as err_mapper:
                result = ResultFailure(ok=False, error=str(err_mapper), issues={Issue.IMAGE_BATCH_ERROR})
            except CRSInputError as err_crs:
                result = ResultFailure(ok=False, error=str(err_crs), issues={Issue.IMAGE_BATCH_ERROR})
            except CoordinateTransformationError as err_trafo:
                result = ResultFailure(ok=False, error=str(err_trafo), issues={Issue.IMAGE_BATCH_ERROR})
            except MappingError as err:
                result = ResultFailure(ok=False, error=str(err), issues={Issue.IMAGE_BATCH_ERROR})

            center_p_dict[key] = result

            if result.ok is False:
                all_mapping_worked = False
            # else:
            #    one_mapping_worked = True

        # if not one_mapping_worked:
        #    raise MappingError("For none of the images the center point could be mapped")

        if not all_mapping_worked:
            logger.warning("Some of the center points could not me mapped")

        return center_p_dict

    def map_footprint(
        self,
        points_per_edge: int = 0,
        mapper: MappingBase | None = None,
        transformer: CoordinateTransformer | None = None,
    ) -> dict[str, MappingResult]:
        """Map footprint of images of batch to 3d

        :param points_per_edge: The number of points which should be inserted between corners.
            0: only corner points are mapped
        :param mapper: Optional specify another Mapper to be used,
            can be different to the one assigned in the image class
        :param transformer: A CoordinateTransfoer object which can be passed instead using crs_s
        :return: MappingResult for all images
        :raises ValueError: Image Batch is empty
        :raises NotGeoreferencedError: If an image is not geo-referenced
        :raises MapperMissingError: If no mapper is available
        :raises CRSInputError: If CRS/transformer input is invalid
        :raises CoordinateTransformationError: If coordinate transformation fails
        :raises MappingError: If mapping fails
        """

        footprint_dict = {}

        if self.__len__() == 0:
            raise ValueError("ImageBatch is empty")

        # one_mapping_worked = False
        all_mapping_worked = True
        for key, image in self.images.items():
            try:
                result = image.map_footprint(
                    points_per_edge=points_per_edge,
                    mapper=self.mapper if mapper is None else mapper,
                    transformer=transformer,
                )
            except NotGeoreferencedError as err_georef:
                result = ResultFailure(ok=False, error=str(err_georef), issues={Issue.IMAGE_BATCH_ERROR})
            except MapperMissingError as err_mapper:
                result = ResultFailure(ok=False, error=str(err_mapper), issues={Issue.IMAGE_BATCH_ERROR})
            except CRSInputError as err_crs:
                result = ResultFailure(ok=False, error=str(err_crs), issues={Issue.IMAGE_BATCH_ERROR})
            except CoordinateTransformationError as err_trafo:
                result = ResultFailure(ok=False, error=str(err_trafo), issues={Issue.IMAGE_BATCH_ERROR})
            except MappingError as err:
                result = ResultFailure(ok=False, error=str(err), issues={Issue.IMAGE_BATCH_ERROR})

            footprint_dict[key] = result

            if result.ok is False:
                all_mapping_worked = False
            # else:
            #    one_mapping_worked = True

        # if not one_mapping_worked:
        # raise MappingError("For none of the images a footprint could be mapped")

        if not all_mapping_worked:
            logger.warning("Some of the footprints could not me mapped")

        return footprint_dict

    def project(
        self,
        coordinates: ArrayNx3,
        crs_s: CRS | None = None,
        transformer: CoordinateTransformer | None = None,
        only_valid: bool = False,
    ) -> dict[str, ProjectionResult] | None:
        """Calculate projection of 3d points into image. If points was outsize image size and to_distortion is true
        the undistorted projection will be returned

        :param coordinates: Array of (nx3) with the point positions
        :type coordinates: ArrayNx3
        :param crs_s: The coordinate system of the input coordinates
        :param transformer: A CoordinateTransfoer object which can be passed instead using crs_s
        :param only_valid: If true only return images with valid projections.
        :return: dict[ImageName, ProjectionResult] or None if for filtered (only_valid) no projections are valid
        :raises ValueError: If the image batch is empty.
        :raises CRSInputError: If both ``crs_s`` and ``transformer`` are provided.
        :raises NotGeoreferencedError: If an image is not geo-referenced.
        :raises MapperMissingError: If a required mapper is missing.
        :raises CoordinateTransformationError: If coordinate transformation fails.
        """

        if self.__len__() == 0:
            raise ValueError("ImageBatch is empty")

        _coordinates = to_array_nx3(coordinates)

        if crs_s is not None and transformer is not None:
            raise CRSInputError("Either crs or transformation can be used or both None")

        projected_dict: dict[str, ProjectionResult] = {}

        for key, image in self.images.items():
            try:
                result = image.project(_coordinates, crs_s, transformer=transformer)

            except NotGeoreferencedError as err_georef:
                result = ResultFailure(ok=False, error=str(err_georef), issues={Issue.IMAGE_BATCH_ERROR})
            except MapperMissingError as err_mapper:
                result = ResultFailure(ok=False, error=str(err_mapper), issues={Issue.IMAGE_BATCH_ERROR})
            except CRSInputError as err_crs:
                result = ResultFailure(ok=False, error=str(err_crs), issues={Issue.IMAGE_BATCH_ERROR})
            except CoordinateTransformationError as err_trafo:
                result = ResultFailure(ok=False, error=str(err_trafo), issues={Issue.IMAGE_BATCH_ERROR})

            projected_dict[key] = result

        if only_valid:
            filtered_dict: dict[str, ProjectionResult] = {}
            for key, value in projected_dict.items():
                if value.ok is True:
                    if np.any(value.mask):
                        filtered_dict[key] = value

            if len(filtered_dict) == 0:
                return None

            return filtered_dict

        return projected_dict
