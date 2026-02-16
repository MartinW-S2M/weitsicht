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


"""Estimate camera intrinsics (IOR) from resolved metadata tags.

This module operates on weitsicht's tag dataclasses
(e.g. :class:`~weitsicht.metadata.tag_systems.tag_base.MetaTagIORBase`)
and is independent of how the metadata was extracted (ExifTool, PIL, custom parsers, ...).
"""

import logging
import math

from weitsicht.camera.opencv_perspective import CameraOpenCVPerspective
from weitsicht.metadata.camera_alternative_tags import AlternativeCalibrationTags
from weitsicht.metadata.camera_database import get_sensor_from_database
from weitsicht.metadata.metadata_results import (
    IORFromMetaResult,
    IORFromMetaResultSuccess,
    MetadataIssue,
)
from weitsicht.metadata.tag_systems.tag_base import MetaTagIORBase, MetaTagIORExtended
from weitsicht.utils import ResultFailure

# Unit conversion factor
inch_to_mm = 25.4
cm_to_mm = 10
um_to_mm = 0.001

logger = logging.getLogger(__name__)


def get_unit_factor(resolution_unit) -> float | None:
    """Factor to scale the exif resolution unit to millimeter which is used for Focal length in EXIF.
    Tag 0xa20e - FocalPlaneXResolution
    Tag 0xa210 - FocalPlaneResolutionUnit
    https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/EXIF.html
    We assume square Pixels, so we only will do it for the image width side

    :param resolution_unit: the resolution unit value given in the EXIF
    :return: Unit factor or None
    """
    if resolution_unit == 2:  # Inch
        return inch_to_mm
    elif resolution_unit == 3:  # Centimeter
        return cm_to_mm
    elif resolution_unit == 4:  # Millimeter
        return 1
    elif resolution_unit == 5:  # Micrometer
        return um_to_mm
    else:
        return None


def compute_sensor_width_in_mm(image_width: int, tags: MetaTagIORBase) -> float | None:
    """Get Sensor with in mm using the image width in pixel and the tags ResolutionUnit and FocalPlaneXResolution.

    :param image_width: Width of image in Pixels
    :param tags: resolved metadata values
    :returns: Sensor width in mm or None if not possible"""

    resolution_unit = tags.focal_plane_resolution_unit
    pixels_per_unit = tags.focal_plane_x_resolution
    pixels_per_unit_y = tags.focal_plane_y_resolution

    if resolution_unit is None or pixels_per_unit is None:
        return None

    unit_factor = get_unit_factor(resolution_unit)

    if not unit_factor:
        return None

    if pixels_per_unit <= 0.0:
        # Some wrongly formatted camera have negative resolutions
        # We check if at least YResolution is present and if not negative use that
        if pixels_per_unit_y is None or pixels_per_unit_y <= 0.0:
            return None
        pixels_per_unit = pixels_per_unit_y

    pixel_pitch = unit_factor / pixels_per_unit

    return image_width * pixel_pitch


def compute_focal_length_from_35mm(
    focal_35mm: float, image_width: int | float, image_height: int | float
) -> float | None:
    # If a focal length of 35mm exists together with image width this is a good approximating
    if focal_35mm > 0:
        mm_to_pixel = 43.3 / math.sqrt(image_width**2 + image_height**2)  # 35mm film have a sensor size of 36x24mm.
        focal_pixel = focal_35mm / mm_to_pixel
        return focal_pixel
    return None


def compute_from_sensor_width(focal_mm: float, sensor_width: float, image_width: int) -> float:
    return focal_mm / (sensor_width / image_width)


def estimate_camera(tags: MetaTagIORBase) -> tuple[int, int, float, float, float]:
    """Estimate the standard camera parameters.
    Even though I am not sure if 35mm equivalent is always calculated correctly we will use that as first source,
    because for resampled images there could be that original exif-tags are left (e.g. PlaneResolution for example)
    which would be wrong for the resampled image to have correct focal length in pixel.
    Currently, 4 possible ways are implemented
    (1) Using 35mm equivalent
    (2) Using focal length and Resolution unit
    (3) Using focal length and Sensor Database; this works only if camera type is present in database
    More advanced calibration tags in XMP like that one from Pix4D will be treated separately
    The principal point is for that base estimation the center of the image

    :param meta_data: Metadata dict following exif-tools tags
    :raises ValueError: If standard Tags can not be used or if dimensions are not given
    :returns: tuple(width, height, focal length in pixel, c_x(principal point), c_y(principal point))"""

    # Get dimensions of image
    width = int(tags.image_shape[0])
    height = int(tags.image_shape[1])

    if width == 0 or height == 0:
        raise ValueError("No image dimension are present in meta data")

    # FocalLength is the real focal length in mm
    focal = tags.focal_length or 0.0

    # Focal_35mm is the focal length if the camera would be a 35mm Format camera to have the same field of view
    # If focal_35 is present it is always possible to derive focal length in pixel
    # Therefore we try this first as then we are always able to create a valid camera class
    # But as it turns out, not all vendors are very precise how this is derived compared to the focal length
    focal_35 = tags.focal_length_35mm or 0.0

    focal_pixel = None

    # 1st try - get focal length in pixel from 35mm equivalent
    if focal_35 > 0:
        focal_pixel = compute_focal_length_from_35mm(focal_35, width, height)
        logger.debug(f"Estimated Focal Length in pixel from 35mm equivalent: f={focal_pixel:6.2f}")

    # 2nd try  - estimate focal length in pixel from the tag focalLength
    # We will always try this as well even if focal_35 is present
    if focal > 0.0:
        sensor_width = compute_sensor_width_in_mm(width, tags)
        if sensor_width is not None:
            focal_pixel = compute_from_sensor_width(focal, sensor_width=sensor_width, image_width=width)
            logger.debug(f"Estimated Focal Length in pixel from sensor with: f={focal_pixel:6.2f}")

    # Principal point default will be center of image size
    # Later we try if in the metadata are better values
    c_x = width / 2.0
    c_y = height / 2.0

    # 3rd try - We check other tags from different software packages and drone vendors
    # E.g. newer DJI model save calibrated values CalibratedFocalLength, CalibratedOpticalCenterX/Y
    # Or Pix4D has specified a tag system for the IOR and also for coordinate systems
    # CalibratedFocalLength: 3666.666504
    # CalibratedOpticalCenterX: 2736.000000

    # We use a wrapper to call all different tag systems
    # res = calibration_estimator(meta_data)
    # if res is not None:
    #    width, height, focal_pixel, c_x, c_y, distortion = res

    # 4th and last try - estimate focal length from sensor database -
    # No sensor size is available in metadata
    # This is highly depending on the camera database and if the sensor is available there
    # If the sensor is not found you can add it in "camare_database.py"
    # Anyhow we will try this as this might be more accurate then estimating sensor size from exif data
    if focal > 0.0:
        sensor_size = get_sensor_from_database(make=tags.make, model=tags.model)
        if sensor_size is not None:
            focal_pixel = compute_from_sensor_width(focal, sensor_width=sensor_size[0], image_width=width)
            # logger.debug("Estimated Focal Length in pixel from sensor with from database: f=%6.2f" % focal_pixel)

    if focal_pixel is None:
        raise ValueError("Estimation of focal length in pixel failed")
    return width, height, focal_pixel, c_x, c_y


def ior_from_meta(tags_ior: MetaTagIORBase, tags_ior_extended: MetaTagIORExtended) -> IORFromMetaResult:
    """Estimate camera intrinsics (IOR) from metadata tags.

    :param tags_ior: Standard intrinsic-related tags.
    :type tags_ior: MetaTagIORBase
    :param tags_ior_extended: Optional extended calibration tags.
    :type tags_ior_extended: MetaTagIORExtended
    :return: Successful IOR result or a failure result.
    :rtype: IORFromMetaResult
    """
    try:
        width, height, focal_pixel, c_x, c_y = estimate_camera(tags=tags_ior)
    except ValueError as err:
        return ResultFailure(ok=False, error=str(err), issues={MetadataIssue.IOR_FAILED})

    # This is the model which should always work with minimum meta data present
    # At least for standard digital camera images
    # If we would start to use other camera systems like fishey models we would need to decide here which one to use
    camera = CameraOpenCVPerspective(width=width, height=height, fx=focal_pixel, fy=focal_pixel, cx=c_x, cy=c_y)

    # try to find better camera model in metadata (EXIF/XMP)
    cam_result = AlternativeCalibrationTags.run_all_methods(tags_ior=tags_ior, tags_ior_extended=tags_ior_extended)

    # It the result was not None we use that specific camera calibration, otherwise we use the standard one
    if cam_result is not None:
        camera = cam_result

    return IORFromMetaResultSuccess(ok=True, camera=camera, width=width, height=height)
