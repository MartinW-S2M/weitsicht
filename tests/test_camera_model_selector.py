from pathlib import Path

import pytest

from weitsicht.camera.camera_dict_selector import get_camera_from_dict

DATA_DIR = Path(__file__).parent.resolve() / "data"


def test_select_camera_from_dict():

    cam_param_dict = {
        "type": "OpenCV",
        "calib_width": 8256,
        "calib_height": 5504,
        "fx": 4678.567517082265,
        "fy": 4678.567517082265,
        "cx": 4199.726251723716,
        "cy": 2749.841800528425,
        "k1": -0.043081934066152315,
        "k2": 0.013751556689262834,
        "k3": 0.0,
        "k4": 0.0,
        "p1": -0.0007525372490219161,
        "p2": 0.0027883695798851496,
    }

    cam = get_camera_from_dict(cam_param_dict)

    assert cam.type.fullname == "OpenCV"

    # wrong dict
    wrong_dict = {}
    with pytest.raises(KeyError):
        get_camera_from_dict(wrong_dict)
