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
from pyproj import CRS, Proj
from pyproj.crs.crs import CompoundCRS

from weitsicht.exceptions import CoordinateTransformationError
from weitsicht.metadata.metadata_results import (
    EORFromMetaResult,
    EORFromMetaResultSuccess,
    MetadataIssue,
)
from weitsicht.metadata.tag_systems.tag_base import MetaTagAll
from weitsicht.transform.rotation import Rotation
from weitsicht.transform.utm_converter import point_convert_utm_wgs84_egm2008, point_wgs84ell_to_utm
from weitsicht.transform.wgs84_local_tangent import WGS84LocalTangent
from weitsicht.utils import Array3x3, ResultFailure

logger = logging.getLogger(__name__)

r_body_to_cam_down_facing = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
r_body_to_cam_in_x_facing = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])


def _deg(val):
    return float(val) * np.pi / 180.0


def rot_body_ned_from_meta(tags: MetaTagAll) -> tuple[Array3x3, bool] | None:
    # Orientation normally in NED and aircraft convention
    # Some of the tags do actually refere to the image direction of view
    # Others refere to angles of the aircraft and the camera is mounted looking down

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
        return None

    if pitch is None:
        return None

    # Rotation of IMAGE still in Body System assume NED
    rot_ned_body = np.array(
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

    return rot_ned_body, angle_in_direction_of_view


def eor_from_meta(
    tags: MetaTagAll,
    crs: CRS | None = None,
    vertical_ref: str = "ellipsoidal",
    height_rel: float = 0.0,
    to_utm: bool = False,
) -> EORFromMetaResult:
    """Extract exterior orientation (position + rotation + CRS) from metadata tags.

    If ``to_utm`` is ``True`` the returned position is expressed in a local projected coordinate system
    derived from the input CRS (typically WGS84) by converting to WGS84-UTM with EGM2008 heights.
    The orientation is returned in the corresponding UTM grid ENU frame (i.e. true ENU rotated by
    meridian convergence).

    If ``to_utm`` is ``False`` the pose is returned in WGS84 ECEF (EPSG:4978).

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
    :param to_utm: Whether to output the position in WGS84-UTM (EGM2008) instead of ECEF, defaults to ``False``.
    :type to_utm: bool
    :return: Successful EOR result or a failure result.
    :rtype: EORFromMetaResult
    """

    crs_source: CRS | None = crs

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

    # First we will check if we are using relative heights
    # The problem is that in old DJI systems the altitude was very bad
    # But the reference for the relative height was not stored in the meta data
    # It was often the take-off point but that information is not given normaly if you have only images.
    # Thus we have that parameter height_rel which states the height the rel_altitude referes to.
    rel_z_exif = None
    if vertical_ref == "relative":
        rel_z_exif = tags.z_alternatives.rel_altitude
        if rel_z_exif is not None:
            z_exif = height_rel + float(rel_z_exif)

    # Determine CRS for interpreting the input coordinates if not overridden by the user.
    is_wgs84 = False
    if crs_source is None:
        # First this will be standard based on if vertical ref is specified
        if vertical_ref == "orthometric":
            crs_hor_exif = 4326
            crs_vert_exif = 3855

        else:
            crs_hor_exif = tags.crs.horiz_cs or 4979
            crs_vert_exif = tags.crs.vert_cs or "ellipsoidal"

        # Just to be sure, this maybe interfere with vertical ref
        # TODO this could overrite the vertical_ref specified
        if tags.crs.horiz_cs is not None:
            crs_hor_exif = tags.crs.horiz_cs
            crs_vert_exif = tags.crs.vert_cs or "ellipsoidal"

        # This is now an override if relative height is specified:
        # Then we will use 4326+3855
        if rel_z_exif is not None:
            crs_hor_exif = 4326
            crs_vert_exif = 3855

        # Create now the CRS
        if crs_vert_exif == "ellipsoidal":
            crs_source = CRS(crs_hor_exif).to_3d()
        else:
            crs_hor = CRS.from_user_input(crs_hor_exif)
            crs_vert = CRS.from_user_input(crs_vert_exif)
            crs_source = CompoundCRS(f"{crs_hor_exif}+{crs_vert_exif}", [crs_hor, crs_vert])

        is_wgs84 = True
    # We assume that even if the position is given in another CRS,
    # still the angles are that one of the local tangent plane at this point
    # Therefore we transform only the position if a crs is given as parameter
    # Transform that to ECEF and make the NED to ECEF conversion

    assert crs_source is not None

    if is_wgs84:
        ltp_frame = WGS84LocalTangent.from_wgs84ell_crs(
            crs_s=crs_source, lon_deg=x_exif, lat_deg=y_exif, h_m=z_exif, skip_ecef=to_utm
        )
    else:
        ltp_frame = WGS84LocalTangent.from_crs(x=x_exif, y=y_exif, z=z_exif, crs_s=crs_source)

    # GET the rotation of the iamge in that case UAV/drone
    # It is supposed to be in NED and axis follow aircraft convention8
    result_rot = rot_body_ned_from_meta(tags=tags)

    if result_rot is None:
        return ResultFailure(
            ok=False,
            error="Orientation angles are not in meta-data",
            issues={MetadataIssue.MISSING_ORIENTATION, MetadataIssue.EOR_FAILED},
        )

    rot_body_ned, angle_in_direction_of_view = result_rot
    rot_ecef_body = ltp_frame.to_ecef_matrix(rot_body_ned, local_frame="NED")

    # We have now the rotation in ecef
    # Need to convert to allign with camera axis from body axis
    if angle_in_direction_of_view:
        rot_ecef_cam = rot_ecef_body @ r_body_to_cam_in_x_facing
    else:
        rot_ecef_cam = rot_ecef_body @ r_body_to_cam_down_facing
    if to_utm is False:
        return EORFromMetaResultSuccess(
            ok=True,
            position=ltp_frame.origin_ecef,
            orientation=Rotation(rot_ecef_cam),
            crs=ltp_frame.crs_ecef,
        )

    # FOR UTM projection to calculate we go directly from ell coordinates
    # So if we already have WGS84 ell coordiantes we can go directly to UTM
    # Route for projected output: position in UTM, orientation in UTM referencing to grid north.

    lon_deg, lat_deg, h_m = ltp_frame.origin_ell[0], ltp_frame.origin_ell[1], ltp_frame.origin_ell[2]

    try:
        if is_wgs84:
            x, y, z, crs_result = point_wgs84ell_to_utm(crs_source, lon_deg, lat_deg, h_m)
        else:
            origin_ecef = ltp_frame.origin_ecef
            x, y, z, crs_result = point_convert_utm_wgs84_egm2008(
                crs_source, origin_ecef[0], origin_ecef[1], origin_ecef[2]
            )
    except (ValueError, CoordinateTransformationError) as err:
        return ResultFailure(
            ok=False, error=str(err), issues={MetadataIssue.TRANSFORMATION_FAILED, MetadataIssue.EOR_FAILED}
        )

    position = np.array([x, y, z])

    # FOR UTM the Rotation Matrix in ENU (true north)
    rot_enu_cam = ltp_frame.to_ltp_matrix(rot_ecef_cam, local_frame="ENU")

    # Apply grid convergence so that the returned orientation matches the UTM grid north.
    crs_utm = crs_result.sub_crs_list[0] if crs_result.sub_crs_list else crs_result
    convergence_deg = Proj(crs_utm).get_factors(lon_deg, lat_deg).meridian_convergence
    g = float(np.deg2rad(convergence_deg))
    rz_gamma = np.array(
        [
            [np.cos(g), -np.sin(g), 0.0],
            [np.sin(g), np.cos(g), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rot_utm_cam = rz_gamma @ rot_enu_cam

    return EORFromMetaResultSuccess(ok=True, position=position, orientation=Rotation(rot_utm_cam), crs=crs_result)
