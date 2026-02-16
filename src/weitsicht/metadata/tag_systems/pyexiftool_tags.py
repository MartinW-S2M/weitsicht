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

from collections.abc import Mapping
from typing import Any

from weitsicht.metadata.tag_systems.tag_base import (
    MetaTagCRS,
    MetaTagGPS,
    MetaTagIORBase,
    MetaTagIORExtended,
    MetaTagOrientation,
    MetaTagsBase,
    MetaTagZalternatives,
)


class PyExifToolTags(MetaTagsBase):
    """Using Phil Harvey's exiftool and the wrapper PyExifTool."""

    def __init__(self, meta_data: Mapping[str, Any]):
        self.meta_data = meta_data

    def image_shape(self):

        if "EXIF:ImageWidth" in self.meta_data.keys():
            width = int(self.meta_data["EXIF:ImageWidth"])
            height = int(self.meta_data["EXIF:ImageHeight"])
        elif "EXIF:EXIFImageWidth" in self.meta_data.keys():
            width = int(self.meta_data["EXIF:EXIFImageWidth"])
            height = int(self.meta_data["EXIF:EXIFImageHeight"])
        elif "File:ImageWidth" in self.meta_data.keys():
            width = int(self.meta_data["File:ImageWidth"])
            height = int(self.meta_data["File:ImageHeight"])
        elif "XMP:ImageWidth" in self.meta_data.keys():
            width = int(self.meta_data["XMP:ImageWidth"])
            height = int(self.meta_data["XMP:ImageHeight"])
        else:
            width = 0
            height = 0

        image_shape = (width, height)
        return image_shape

    def get_ior_base(self):

        image_shape = self.image_shape()
        focal_plane_resolution_unit = None
        if "EXIF:FocalPlaneResolutionUnit" in self.meta_data.keys():
            focal_plane_resolution_unit = int(self.meta_data["EXIF:FocalPlaneResolutionUnit"])

        focal_plane_x_resolution = 0.0
        if "EXIF:FocalPlaneXResolution" in self.meta_data.keys():
            focal_plane_x_resolution = float(self.meta_data["EXIF:FocalPlaneXResolution"])

        focal_plane_y_resolution = 0.0
        if "EXIF:FocalPlaneYResolution" in self.meta_data.keys():
            focal_plane_y_resolution = float(self.meta_data["EXIF:FocalPlaneYResolution"])

        make = self.meta_data.get("EXIF:Make", "")
        model = self.meta_data.get("EXIF:Model", "")

        focal_length = 0.0
        if "EXIF:FocalLength" in self.meta_data.keys():
            focal_length = float(self.meta_data["EXIF:FocalLength"])

        focal_length_35mm = 0.0
        if "EXIF:FocalLengthIn35mmFormat" in self.meta_data.keys():
            focal_length_35mm = float(self.meta_data["EXIF:FocalLengthIn35mmFormat"])

        tags = MetaTagIORBase(
            image_shape=image_shape,
            focal_plane_resolution_unit=focal_plane_resolution_unit,
            focal_plane_x_resolution=focal_plane_x_resolution,
            focal_plane_y_resolution=focal_plane_y_resolution,
            make=make,
            model=model,
            focal_length=focal_length,
            focal_length_35mm=focal_length_35mm,
        )
        return tags

    def get_standard_gps(self) -> MetaTagGPS:

        longitude_ref = self.meta_data.get("EXIF:GPSLongitudeRef", "E")
        latitude_ref = self.meta_data.get("EXIF:GPSLatitudeRef", "N")
        longitude = None
        latitude = None
        altitude = None

        if {
            "EXIF:GPSLongitude",
            "EXIF:GPSLatitude",
            "EXIF:GPSAltitude",
        } <= self.meta_data.keys():
            longitude = float(self.meta_data["EXIF:GPSLongitude"])
            latitude = float(self.meta_data["EXIF:GPSLatitude"])
            altitude = float(self.meta_data["EXIF:GPSAltitude"])

        tags = MetaTagGPS(
            gps_latitude=latitude,
            gps_latitude_ref=latitude_ref,
            gps_longitude=longitude,
            gps_longitude_ref=longitude_ref,
            gps_altitude=altitude,
        )

        return tags

    def get_crs(self) -> MetaTagCRS:

        horiz_cs = self.meta_data.get("XMP:HorizCS")
        if horiz_cs is not None:
            horiz_cs = str(horiz_cs).strip() or None

        vert_cs = self.meta_data.get("XMP:VertCS")
        if vert_cs is not None:
            vert_cs = str(vert_cs).strip() or None

        tags = MetaTagCRS(horiz_cs=horiz_cs, vert_cs=vert_cs)

        return tags

    def get_z_alternatives(self) -> MetaTagZalternatives:

        rel_altitude = 0.0
        if "XMP:RelativeAltitude" in self.meta_data.keys():
            rel_altitude = float(self.meta_data["XMP:RelativeAltitude"])

        tags = MetaTagZalternatives(rel_altitude=rel_altitude)

        return tags

    def get_orientation_values(self) -> MetaTagOrientation:

        xmp_camera_pitch = None
        xmp_camera_roll = None
        xmp_camera_yaw = None
        if {
            "XMP:CameraPitch",
            "XMP:CameraRoll",
            "XMP:CameraYaw",
        } <= self.meta_data.keys():
            xmp_camera_pitch = float(self.meta_data["XMP:CameraPitch"])
            xmp_camera_roll = float(self.meta_data["XMP:CameraRoll"])
            xmp_camera_yaw = float(self.meta_data["XMP:CameraYaw"])

        maker_notes_camera_pitch = None
        maker_notes_camera_roll = None
        maker_notes_camera_yaw = None
        if {
            "MakerNotes:CameraPitch",
            "MakerNotes:CameraRoll",
            "MakerNotes:CameraYaw",
        } <= self.meta_data.keys():
            maker_notes_camera_pitch = float(self.meta_data["MakerNotes:CameraPitch"])
            maker_notes_camera_roll = float(self.meta_data["MakerNotes:CameraRoll"])
            maker_notes_camera_yaw = float(self.meta_data["MakerNotes:CameraYaw"])

        xmp_pitch = None
        xmp_roll = None
        xmp_yaw = None
        if {"XMP:Pitch", "XMP:Roll", "XMP:Yaw"} <= self.meta_data.keys():
            xmp_pitch = float(self.meta_data["XMP:Pitch"])
            xmp_roll = float(self.meta_data["XMP:Roll"])
            xmp_yaw = float(self.meta_data["XMP:Yaw"])

        maker_notes_pitch = None
        maker_notes_roll = None
        maker_notes_yaw = None
        if {
            "MakerNotes:Pitch",
            "MakerNotes:Roll",
            "MakerNotes:Yaw",
        } <= self.meta_data.keys():
            maker_notes_pitch = float(self.meta_data["MakerNotes:Pitch"])
            maker_notes_roll = float(self.meta_data["MakerNotes:Roll"])
            maker_notes_yaw = float(self.meta_data["MakerNotes:Yaw"])

        xmp_gimbal_roll_deg = None
        xmp_gimbal_yaw_deg = None
        xmp_gimbal_pitch_deg = None
        if {
            "XMP:GimbalRollDegree",
            "XMP:GimbalYawDegree",
            "XMP:GimbalPitchDegree",
        } <= self.meta_data.keys():
            xmp_gimbal_roll_deg = float(self.meta_data["XMP:GimbalRollDegree"])
            xmp_gimbal_yaw_deg = float(self.meta_data["XMP:GimbalYawDegree"])
            xmp_gimbal_pitch_deg = float(self.meta_data["XMP:GimbalPitchDegree"])

        maker_notes_gimbal_roll_deg = None
        maker_notes_gimbal_yaw_deg = None
        maker_notes_gimbal_pitch_deg = None
        if {
            "MakerNotes:GimbalRollDegree",
            "MakerNotes:GimbalYawDegree",
            "MakerNotes:GimbalPitchDegree",
        } <= self.meta_data.keys():
            maker_notes_gimbal_roll_deg = float(self.meta_data["MakerNotes:GimbalRollDegree"])
            maker_notes_gimbal_yaw_deg = float(self.meta_data["MakerNotes:GimbalYawDegree"])
            maker_notes_gimbal_pitch_deg = float(self.meta_data["MakerNotes:GimbalPitchDegree"])

        tags = MetaTagOrientation(
            xmp_camera_pitch=xmp_camera_pitch,
            xmp_camera_roll=xmp_camera_roll,
            xmp_camera_yaw=xmp_camera_yaw,
            maker_notes_camera_pitch=maker_notes_camera_pitch,
            maker_notes_camera_roll=maker_notes_camera_roll,
            maker_notes_camera_yaw=maker_notes_camera_yaw,
            xmp_pitch=xmp_pitch,
            xmp_roll=xmp_roll,
            xmp_yaw=xmp_yaw,
            maker_notes_pitch=maker_notes_pitch,
            maker_notes_roll=maker_notes_roll,
            maker_notes_yaw=maker_notes_yaw,
            xmp_gimbal_roll_deg=xmp_gimbal_roll_deg,
            xmp_gimbal_yaw_deg=xmp_gimbal_yaw_deg,
            xmp_gimbal_pitch_deg=xmp_gimbal_pitch_deg,
            maker_notes_gimbal_roll_deg=maker_notes_gimbal_roll_deg,
            maker_notes_gimbal_yaw_deg=maker_notes_gimbal_yaw_deg,
            maker_notes_gimbal_pitch_deg=maker_notes_gimbal_pitch_deg,
        )
        return tags

    def get_ior_extended(self) -> MetaTagIORExtended:

        image_shape = self.image_shape()

        calibrated_focal_length = None
        if "XMP:CalibratedFocalLength" in self.meta_data.keys():
            calibrated_focal_length = self.meta_data["XMP:CalibratedFocalLength"]
        elif "EXIF:CalibratedFocalLength" in self.meta_data.keys():
            calibrated_focal_length = self.meta_data["EXIF:CalibratedFocalLength"]

        calibrated_optical_center_x = None
        calibrated_optical_center_y = None
        if {
            "XMP:CalibratedOpticalCenterY",
            "XMP:CalibratedOpticalCenterX",
        } <= self.meta_data.keys():
            calibrated_optical_center_x = self.meta_data["XMP:CalibratedOpticalCenterX"]
            calibrated_optical_center_y = self.meta_data["XMP:CalibratedOpticalCenterY"]
        elif {
            "EXIF:CalibratedOpticalCenterY",
            "EXIF:CalibratedOpticalCenterX",
        } <= self.meta_data.keys():
            calibrated_optical_center_x = self.meta_data["EXIF:CalibratedOpticalCenterX"]
            calibrated_optical_center_y = self.meta_data["EXIF:CalibratedOpticalCenterY"]

        tags = MetaTagIORExtended(
            image_shape=image_shape,
            calibrated_focal_length=calibrated_focal_length,
            calibrated_optical_center_x=calibrated_optical_center_x,
            calibrated_optical_center_y=calibrated_optical_center_y,
        )

        return tags
