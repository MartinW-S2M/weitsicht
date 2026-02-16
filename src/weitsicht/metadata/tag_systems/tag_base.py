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


"""Tag system registry and helpers.

Each tag system lives in its own module with a concrete resolver class.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class MetaTagIORBase:
    image_shape: tuple[int, int]
    focal_plane_resolution_unit: int | None
    focal_plane_x_resolution: float | None
    focal_plane_y_resolution: float | None
    make: str
    model: str
    focal_length: float | None
    focal_length_35mm: float | None


@dataclass
class MetaTagGPS:
    gps_latitude: float | None
    gps_latitude_ref: str | None
    gps_longitude: float | None
    gps_longitude_ref: str | None
    gps_altitude: float | None


@dataclass
class MetaTagCRS:
    horiz_cs: str | None
    vert_cs: str | None


@dataclass
class MetaTagZalternatives:
    rel_altitude: float | None


@dataclass
class MetaTagOrientation:
    xmp_camera_pitch: float | None
    xmp_camera_roll: float | None
    xmp_camera_yaw: float | None

    maker_notes_camera_pitch: float | None
    maker_notes_camera_roll: float | None
    maker_notes_camera_yaw: float | None

    xmp_pitch: float | None
    xmp_roll: float | None
    xmp_yaw: float | None

    maker_notes_pitch: float | None
    maker_notes_roll: float | None
    maker_notes_yaw: float | None

    xmp_gimbal_roll_deg: float | None
    xmp_gimbal_yaw_deg: float | None
    xmp_gimbal_pitch_deg: float | None

    maker_notes_gimbal_roll_deg: float | None
    maker_notes_gimbal_yaw_deg: float | None
    maker_notes_gimbal_pitch_deg: float | None


@dataclass
class MetaTagIORExtended:
    image_shape: tuple[int, int]
    calibrated_focal_length: float | None
    calibrated_optical_center_x: float | None
    calibrated_optical_center_y: float | None


@dataclass
class MetaTagAll:
    """Container holding all resolved tag groups."""

    ior_base: MetaTagIORBase
    ior_extended: MetaTagIORExtended
    gps: MetaTagGPS
    orientation: MetaTagOrientation
    crs: MetaTagCRS
    z_alternatives: MetaTagZalternatives


class MetaTagsBase:
    """Base class how different meta-data parser should look like"""

    def __init__(self, meta_data: Any):
        self.meta_data = meta_data

    @abstractmethod
    def image_shape(self) -> tuple[int, int]:

        return (0, 0)

    @abstractmethod
    def get_ior_base(self) -> MetaTagIORBase:

        tags = MetaTagIORBase(
            image_shape=self.image_shape(),
            focal_plane_resolution_unit=0,
            focal_plane_x_resolution=0,
            focal_plane_y_resolution=0,
            make="",
            model="",
            focal_length=0.0,
            focal_length_35mm=0.0,
        )
        return tags

    @abstractmethod
    def get_standard_gps(self) -> MetaTagGPS:

        tags = MetaTagGPS(
            gps_latitude=None,
            gps_latitude_ref=None,
            gps_longitude=None,
            gps_longitude_ref=None,
            gps_altitude=None,
        )

        return tags

    @abstractmethod
    def get_crs(self) -> MetaTagCRS:
        tags = MetaTagCRS(horiz_cs=None, vert_cs=None)

        return tags

    @abstractmethod
    def get_z_alternatives(self) -> MetaTagZalternatives:
        tags = MetaTagZalternatives(rel_altitude=None)

        return tags

    @abstractmethod
    def get_orientation_values(self) -> MetaTagOrientation:

        tags = MetaTagOrientation(
            xmp_camera_pitch=None,
            xmp_camera_roll=None,
            xmp_camera_yaw=None,
            maker_notes_camera_pitch=None,
            maker_notes_camera_roll=None,
            maker_notes_camera_yaw=None,
            xmp_pitch=None,
            xmp_roll=None,
            xmp_yaw=None,
            maker_notes_pitch=None,
            maker_notes_roll=None,
            maker_notes_yaw=None,
            xmp_gimbal_roll_deg=None,
            xmp_gimbal_yaw_deg=None,
            xmp_gimbal_pitch_deg=None,
            maker_notes_gimbal_roll_deg=None,
            maker_notes_gimbal_yaw_deg=None,
            maker_notes_gimbal_pitch_deg=None,
        )
        return tags

    def get_ior_extended(self) -> MetaTagIORExtended:

        tags = MetaTagIORExtended(
            image_shape=self.image_shape(),
            calibrated_focal_length=None,
            calibrated_optical_center_x=None,
            calibrated_optical_center_y=None,
        )

        return tags

    def get_all(self) -> MetaTagAll:

        tags = MetaTagAll(
            ior_base=self.get_ior_base(),
            ior_extended=self.get_ior_extended(),
            gps=self.get_standard_gps(),
            orientation=self.get_orientation_values(),
            z_alternatives=self.get_z_alternatives(),
            crs=self.get_crs(),
        )

        return tags
