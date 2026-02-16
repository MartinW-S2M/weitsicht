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

import inspect
import logging

from weitsicht.camera.base_perspective import CameraBasePerspective
from weitsicht.camera.opencv_perspective import CameraOpenCVPerspective
from weitsicht.metadata.tag_systems.tag_base import MetaTagIORBase, MetaTagIORExtended

logger = logging.getLogger(__name__)


class AlternativeCalibrationTags:
    """Class which holds function to estimate camera calibration of different types using metadata
    All methods implemented will be called at once. So if you want to extend different tagging systems
    from different software packages or vendors just add a new method

    This is only for the camera calibration aka IOR. Exterior orientation aka EOR will be treated in another class
    This should be unique. All methods will be tried, the first one which is not returning None is used.
    """

    def __init__(self, tags: MetaTagIORBase, tags_ior_extended: MetaTagIORExtended):
        self.tags_ior = tags
        self.tags_ior_extended = tags_ior_extended
        self.width = int(self.tags_ior.image_shape[0])
        self.height = int(self.tags_ior.image_shape[1])

    def dji_tags(self) -> CameraOpenCVPerspective | None:

        focal_pixel = None
        c_y = self.height / 2.0
        c_x = self.width / 2.0
        focal_val = self.tags_ior_extended.calibrated_focal_length
        if focal_val is not None:
            focal_pixel = float(focal_val)

        # For some DjI images calibrated tags have been found
        c_x_val = self.tags_ior_extended.calibrated_optical_center_x
        c_y_val = self.tags_ior_extended.calibrated_optical_center_y
        if c_x_val is not None and c_y_val is not None:
            c_x = float(c_x_val)
            c_y = float(c_y_val)

        # Focal Pixel is not found than do not use that even if Calibrated Center would be found
        if focal_pixel is None:
            return None

        camera = CameraOpenCVPerspective(
            width=self.width,
            height=self.height,
            fx=focal_pixel,
            fy=focal_pixel,
            cx=c_x,
            cy=c_y,
        )
        return camera

    def pix4d_tags(self):

        # TODO implement as described in https://support.pix4d.com/hc/en-us/articles/205732309
        return None

    @staticmethod
    def run_all_methods(tags_ior: MetaTagIORBase, tags_ior_extended: MetaTagIORExtended) -> CameraBasePerspective | None:

        v = AlternativeCalibrationTags(tags_ior, tags_ior_extended)
        attrs = (getattr(v, name) for name in dir(v) if not name.startswith("_"))
        methods = filter(inspect.ismethod, attrs)

        cam_result = None
        for method in methods:
            # print(method)
            try:
                cam_result = method()
                if cam_result is not None:
                    break
            except Exception:
                logger.warning("One of the Calibration Methods throws an exception")
                pass

        return cam_result
