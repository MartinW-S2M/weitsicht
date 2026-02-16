import json
from pathlib import Path
from typing import cast

from weitsicht import CameraOpenCVPerspective, CameraType, PyExifToolTags, ior_from_meta

DATA_DIR = Path(__file__).parent.resolve() / "data"


def test_estimate_camera_from_meta():

    meta_tag_dict = json.load(open(DATA_DIR / "image_2_exiftool_tags.json"))

    tags = PyExifToolTags(meta_data=meta_tag_dict)
    result = ior_from_meta(tags_ior=tags.get_ior_base(), tags_ior_extended=tags.get_ior_extended())

    assert result.ok is True

    assert (result.width, result.height) == (11648, 8736)
    assert result.camera is not None

    # This dict has DJI calibration info
    meta_tag_dict = json.load(open(DATA_DIR / "image_3_exiftool_tags.json"))
    tags = PyExifToolTags(meta_data=meta_tag_dict)
    result = ior_from_meta(tags_ior=tags.get_ior_base(), tags_ior_extended=tags.get_ior_extended())

    assert result.ok is True

    assert result.camera.type == CameraType.OpenCV
    assert isinstance(result.camera, CameraOpenCVPerspective)
    camera = cast(CameraOpenCVPerspective, result.camera)
    assert (result.width, result.height) == (5472, 3648)
    assert camera is not None
    assert camera.focal_length_for_gsd_in_pixel == 3666.666504
    assert camera.cy == 1824.0
