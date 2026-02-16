import json
from pathlib import Path

from weitsicht import PyExifToolTags, ior_from_meta, is_opencv_camera
from weitsicht.metadata.camera_estimator_metadata import estimate_camera

DATA_DIR = Path(__file__).parent.resolve() / "data"


def test_estimate_camera():
    exif_tags = json.load(open(DATA_DIR / "image_exiftool_tags.json"))

    tags = PyExifToolTags(exif_tags)

    cam = estimate_camera(tags=tags.get_ior_base())
    assert cam is not None

    assert cam[0] == 8192
    assert cam[1] == 5460
    assert cam[2] > 0.0

    cam = ior_from_meta(tags_ior=tags.get_ior_base(), tags_ior_extended=tags.get_ior_extended())
    assert cam.ok is True

    exif_tags = json.load(open(DATA_DIR / "image_2_exiftool_tags.json"))

    tags = PyExifToolTags(exif_tags)

    cam2_values = estimate_camera(tags=tags.get_ior_base())

    assert cam2_values is not None

    assert cam2_values[0] == 11648
    assert cam2_values[1] == 8736
    assert cam2_values[2] > 0.0

    cam2 = ior_from_meta(tags_ior=tags.get_ior_base(), tags_ior_extended=tags.get_ior_extended())
    assert cam2.ok is True

    exif_tags = json.load(open(DATA_DIR / "image_3_exiftool_tags.json"))
    tags = PyExifToolTags(exif_tags)
    result = ior_from_meta(tags_ior=tags.get_ior_base(), tags_ior_extended=tags.get_ior_extended())
    assert result.ok is True

    camera = result.camera
    assert is_opencv_camera(camera)
    if is_opencv_camera(camera):
        assert camera.calibration_width == 5472
        assert camera.calibration_height == 3648
        assert camera.fx == 3666.666504
        assert camera.cx == 2736.0
        assert camera.cy == 1824.0
