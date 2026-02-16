import pytest

from weitsicht.image.image_dict_selector import get_image_from_dict


def test_get_image_from_dict():
    # wrong dictionary
    with pytest.raises(KeyError):
        get_image_from_dict({})

    with pytest.raises(ValueError):
        get_image_from_dict({"type": "fisheye"})

    cam = {
        "type": "OpenCV",
        "calib_width": 1000,
        "calib_height": 600,
        "fx": 1000.0,
        "fy": 1000,
    }
    dict_load = {
        "type": "perspective",
        "width": 1000,
        "height": 600,
        "position": [1000.0, 2000.0, 0.0],
        "orientation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "camera": cam,
    }

    image = get_image_from_dict(dict_load)

    assert image is not None

    dict_load_ortho = {
        "type": "ortho",
        "width": 1667,
        "height": 1667,
        "geo_transform": {
            "a": 0.29994001199760045,
            "b": 0.0,
            "c": -1750.0,
            "d": 0.0,
            "e": -0.29994001199760045,
            "f": 338800.0,
            "g": 0.0,
            "h": 0.0,
            "i": 1.0,
        },
        "resolution": 0.29994001199760045,
        "crs": None,
    }
    image = get_image_from_dict(dict_load_ortho)
    assert image is not None
