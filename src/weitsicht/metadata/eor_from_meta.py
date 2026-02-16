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

# We follow the specifications from here, but there is nothing standardized at all
# https://support.pix4d.com/hc/en-us/articles/360016450032

import logging

import numpy as np
from numpy import cos, sin
from pyproj import CRS
from pyproj.crs.crs import CompoundCRS

from weitsicht.exceptions import CoordinateTransformationError
from weitsicht.metadata.metadata_results import (
    EORFromMetaResult,
    EORFromMetaResultSuccess,
    MetadataIssue,
)
from weitsicht.metadata.tag_systems.tag_base import MetaTagAll
from weitsicht.transform.rotation import Rotation
from weitsicht.transform.utm_converter import point_convert_utm_wgs84_egm2008
from weitsicht.utils import ResultFailure

logger = logging.getLogger(__name__)

aircraft_notation_to_front_notation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
swap_ned_to_enu_coo_system = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
swap_body_cam = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
swap_body_cam_gimbal = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])


def _deg(val):
    return float(val) * np.pi / 180.0


def eor_from_meta(
    tags: MetaTagAll,
    crs: CRS | None = None,
    vertical_ref: str = "ellipsoidal",
    height_rel: float = 0.0,
) -> EORFromMetaResult:
    """Extract exterior orientation (position + rotation + CRS) from metadata tags.

    The returned position is expressed in a local projected coordinate system derived from the
    input CRS (typically WGS84) by converting to WGS84-UTM with EGM2008 heights.

    If XMP tags ``HorizCS``/``VertCS`` are missing, defaults to WGS84 (EPSG:4979) with ellipsoidal heights.

    :param tags: Grouped metadata values (e.g. from ``MetaTagsBase.get_all()``).
    :type tags: MetaTagAll
    :param crs: Optional override CRS for interpreting the position tags, defaults to ``None``.
    :type crs: CRS | None
    :param vertical_ref: Vertical reference mode: ``ellipsoidal``, ``orthometric``, or ``relative``.
        ``relative`` uses ``RelativeAltitude`` + ``height_rel``.
    :type vertical_ref: str
    :param height_rel: Reference height (meters) for ``vertical_ref='relative'``, defaults to ``0.0``.
    :type height_rel: float
    :return: Successful EOR result or a failure result.
    :rtype: EORFromMetaResult
    """

    crs_source: CRS | None = crs
    position = np.array([0, 0, 0])

    lon = tags.gps.gps_longitude
    lat = tags.gps.gps_latitude
    alt = tags.gps.gps_altitude
    lon_ref = tags.gps.gps_longitude_ref or "E"
    lat_ref = tags.gps.gps_latitude_ref or "N"

    if lon is not None and lat is not None and alt is not None:
        x_exif = lon
        if lon_ref == "W":
            x_exif = -x_exif
        y_exif = lat
        if lat_ref == "S":
            y_exif = -y_exif
        z_exif = alt
    else:
        return ResultFailure(ok=False, error="Standard GPS tags are missing", issues={MetadataIssue.MISSING_GPS})

    #
    rel_z_exif = None
    if vertical_ref == "relative":
        rel_z_exif = tags.z_alternatives.rel_altitude
        if rel_z_exif is not None:
            z_exif = height_rel + float(rel_z_exif)

    # Determine CRS for interpreting the input coordinates if not overridden by the user.
    if crs_source is None:
        if vertical_ref == "orthometric":
            crs_hor_exif = 4326
            crs_vert_exif = 3855

        else:
            crs_hor_exif = tags.crs.horiz_cs or 4979
            crs_vert_exif = tags.crs.vert_cs or "ellipsoidal"

        if tags.crs.horiz_cs is not None:
            crs_hor_exif = tags.crs.horiz_cs
            crs_vert_exif = tags.crs.vert_cs or "ellipsoidal"

        # This is now an override if relative height is specified:
        # Then we will
        if rel_z_exif is not None:
            crs_hor_exif = 4326
            crs_vert_exif = 3855

        if crs_vert_exif == "ellipsoidal":
            crs_source = CRS(crs_hor_exif).to_3d()
        else:
            crs_source = CompoundCRS(
                str(crs_hor_exif) + "+" + str(crs_vert_exif),
                [crs_hor_exif, crs_vert_exif],
            )

    assert crs_source is not None
    try:
        x, y, z, crs_result = point_convert_utm_wgs84_egm2008(crs_source, x_exif, y_exif, z_exif)
    except (ValueError, CoordinateTransformationError) as err:
        return ResultFailure(
            ok=False, error=str(err), issues={MetadataIssue.TRANSFORMATION_FAILED, MetadataIssue.EOR_FAILED}
        )

    position = np.array([x, y, z])

    # Orientation

    xmp_cam_pitch = tags.orientation.xmp_camera_pitch
    xmp_cam_roll = tags.orientation.xmp_camera_roll
    xmp_cam_yaw = tags.orientation.xmp_camera_yaw

    maker_notes_cam_pitch = tags.orientation.maker_notes_camera_pitch
    maker_notes_cam_roll = tags.orientation.maker_notes_camera_roll
    maker_notes_cam_yaw = tags.orientation.maker_notes_camera_yaw

    xmp_pitch_xyz = tags.orientation.xmp_pitch
    xmp_roll_xyz = tags.orientation.xmp_roll
    xmp_yaw_xyz = tags.orientation.xmp_yaw

    maker_pitch_xyz = tags.orientation.maker_notes_pitch
    maker_roll_xyz = tags.orientation.maker_notes_roll
    maker_yaw_xyz = tags.orientation.maker_notes_yaw

    xmp_gimbal_roll = tags.orientation.xmp_gimbal_roll_deg
    xmp_gimbal_yaw = tags.orientation.xmp_gimbal_yaw_deg
    xmp_gimbal_pitch = tags.orientation.xmp_gimbal_pitch_deg

    maker_notes_gimbal_roll = tags.orientation.maker_notes_gimbal_roll_deg
    maker_notes_gimbal_yaw = tags.orientation.maker_notes_gimbal_yaw_deg
    maker_notes_gimbal_pitch = tags.orientation.maker_notes_gimbal_pitch_deg

    angle_in_direction_of_view = False
    pitch = None
    if xmp_cam_roll is not None and xmp_cam_yaw is not None and xmp_cam_pitch is not None:
        angle_in_direction_of_view = True
        roll = _deg(xmp_cam_roll)
        yaw = _deg(xmp_cam_yaw)
        pitch = _deg(xmp_cam_pitch)
    elif maker_notes_cam_roll is not None and maker_notes_cam_yaw is not None and maker_notes_cam_pitch is not None:
        angle_in_direction_of_view = True
        roll = _deg(maker_notes_cam_roll)
        yaw = _deg(maker_notes_cam_yaw)
        pitch = _deg(maker_notes_cam_pitch)
    elif xmp_roll_xyz is not None and xmp_yaw_xyz is not None and xmp_pitch_xyz is not None:
        roll = _deg(xmp_roll_xyz)
        yaw = _deg(xmp_yaw_xyz)
        pitch = _deg(xmp_pitch_xyz)
    elif maker_roll_xyz is not None and maker_yaw_xyz is not None and maker_pitch_xyz is not None:
        roll = _deg(maker_roll_xyz)
        yaw = _deg(maker_yaw_xyz)
        pitch = _deg(maker_pitch_xyz)
    elif xmp_gimbal_roll is not None and xmp_gimbal_yaw is not None and xmp_gimbal_pitch is not None:
        angle_in_direction_of_view = True
        roll = _deg(xmp_gimbal_roll)
        yaw = _deg(xmp_gimbal_yaw)
        pitch = _deg(xmp_gimbal_pitch)
    elif (
        maker_notes_gimbal_roll is not None
        and maker_notes_gimbal_yaw is not None
        and maker_notes_gimbal_pitch is not None
    ):
        angle_in_direction_of_view = True
        roll = _deg(maker_notes_gimbal_roll)
        yaw = _deg(maker_notes_gimbal_yaw)
        pitch = _deg(maker_notes_gimbal_pitch)
    else:
        return ResultFailure(
            ok=False,
            error="Orientation angles are not in meta-data",
            issues={MetadataIssue.MISSING_ORIENTATION, MetadataIssue.EOR_FAILED},
        )

    if pitch is None:
        return ResultFailure(
            ok=False,
            error="Orientation angles are not in meta-data",
            issues={MetadataIssue.MISSING_ORIENTATION, MetadataIssue.EOR_FAILED},
        )
    # Rotation of IMAGE still in Body System
    rot_sys = np.array(
        [
            [
                cos(pitch) * cos(yaw),
                sin(roll) * sin(pitch) * cos(yaw) - cos(roll) * sin(yaw),
                cos(roll) * sin(pitch) * cos(yaw) + sin(roll) * sin(yaw),
            ],
            [
                cos(pitch) * sin(yaw),
                sin(roll) * sin(pitch) * sin(yaw) + cos(roll) * cos(yaw),
                cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw),
            ],
            [-sin(pitch), sin(roll) * cos(pitch), cos(roll) * cos(pitch)],
        ]
    )

    # Bring rotation into the cameras coordinate system. X left, Y top, Z backwards of viewing direction
    rot_enu_body = (swap_ned_to_enu_coo_system @ rot_sys) @ aircraft_notation_to_front_notation

    if angle_in_direction_of_view:
        rot_enu_cam = rot_enu_body @ swap_body_cam_gimbal
    else:
        rot_enu_cam = rot_enu_body @ swap_body_cam

    orientation = Rotation(rot_enu_cam)

    return EORFromMetaResultSuccess(ok=True, position=position, orientation=orientation, crs=crs_result)
