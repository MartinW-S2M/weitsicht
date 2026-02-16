from pathlib import Path

import numpy as np
import pytest

from weitsicht import Issue
from weitsicht.camera.opencv_perspective import CameraOpenCVPerspective
from weitsicht.image.image_batch import ImageBatch
from weitsicht.image.perspective import ImagePerspective
from weitsicht.mapping.horizontal_plane import MappingHorizontalPlane
from weitsicht.mapping.map_trimesh import MappingTrimesh
from weitsicht.transform.rotation import Rotation

DATA_DIR = Path(__file__).parent.resolve() / "data"


@pytest.fixture
def images():
    image_dict = {}

    cam_20mm = CameraOpenCVPerspective(
        width=8256,
        height=5504,
        fx=4678.5675170822651,
        fy=4678.5675170822651,
        cx=8256 / 2.0 + 72.226251723716217 - 0.5,
        cy=5504 / 2.0 - 1.6581994715748369 - 0.5,
        k1=-0.043081934066152315,
        k2=0.013751556689262834,
        p1=-0.00075253724902191611,
        p2=0.0027883695798851496,
    )

    mapper_wall = MappingTrimesh(DATA_DIR / "decimated_160k_31256.ply")

    ext_ori_images = [
        [
            "INDIGO_2021-12-17_Z7II-B_0039",
            2643.0560,
            342715.0911648543551564,
            4.9515,
            -58.407,
            85.239,
            149.603,
        ],
        [
            "INDIGO_2021-12-17_Z7II-B_0040",
            2643.2077,
            342717.0722784474492073,
            4.9169,
            -53.782,
            86.843,
            144.631,
        ],
        [
            "INDIGO_2021-12-17_Z7II-B_0041",
            2643.1326,
            342719.1589016793295741,
            4.9015,
            -20.583,
            83.681,
            110.981,
        ],
        [
            "INDIGO_2021-12-17_Z7II-B_0042",
            2642.9176,
            342720.8983379695564508,
            4.9114,
            -80.291,
            58.317,
            173.197,
        ],
        [
            "INDIGO_2021-12-17_Z7II-B_0043",
            2645.3547,
            342718.4010614864528179,
            5.0342,
            -102.517,
            82.314,
            -166.5470,
        ],
        [
            "INDIGO_2021-12-17_Z7II-B_0044",
            2642.7455,
            342718.9792082067579031,
            5.0487,
            102.283,
            87.869,
            79.249,
        ],
        [
            "INDIGO_2021-12-17_Z7II-B_0045",
            2642.3706,
            342717.4982716664671898,
            5.0628,
            7.087,
            88.263,
            174.190,
        ],
        [
            "INDIGO_2021-12-17_Z7II-B_0046",
            2642.4335,
            342715.4646163340657949,
            5.0715,
            -58.419,
            86.871,
            -121.629,
        ],
        [
            "INDIGO_2021-12-17_Z7II-B_0047",
            2642.5897,
            342713.6570004606619477,
            5.0697,
            -125.144,
            88.884,
            -54.450,
        ],
        [
            "INDIGO_2021-12-17_Z7II-B_0048",
            2642.7148,
            342713.2602515472099185,
            4.9537,
            81.508,
            57.961,
            5.612,
        ],
    ]

    for eor in ext_ori_images:
        position = np.array([eor[1], eor[2], eor[3]])
        orientation = Rotation.from_opk_degree(eor[4], eor[5], eor[6])
        image = ImagePerspective(
            width=8256,
            height=5504,
            position=position,
            orientation=orientation,
            crs=None,
            camera=cam_20mm,
            mapper=mapper_wall,
        )
        image_dict[eor[0]] = image

    images = ImageBatch(image_dict)

    return images


def test_add_images(images):
    image_dict = {}

    cam_20mm = CameraOpenCVPerspective(
        width=8256,
        height=5504,
        fx=4678.5675170822651,
        fy=4678.5675170822651,
        cx=8256 / 2.0 + 72.226251723716217 - 0.5,
        cy=5504 / 2.0 - 1.6581994715748369 - 0.5,
        k1=-0.043081934066152315,
        k2=0.013751556689262834,
        p1=-0.00075253724902191611,
        p2=0.0027883695798851496,
    )

    ext_ori_images = [
        [
            "INDIGO_2021-12-17_Z7II-B_0039",
            2643.0560,
            342715.0911648543551564,
            4.9515,
            -58.407,
            85.239,
            149.603,
        ],
        [
            "INDIGO_2021-12-17_Z7II-B_0040",
            2643.2077,
            342717.0722784474492073,
            4.9169,
            -53.782,
            86.843,
            144.631,
        ],
    ]

    for eor in ext_ori_images:
        position = np.array([eor[1], eor[2], eor[3]])
        orientation = Rotation.from_opk_degree(eor[4], eor[5], eor[6])
        image = ImagePerspective(
            width=8256,
            height=5504,
            position=position,
            orientation=orientation,
            crs=None,
            camera=cam_20mm,
        )
        image_dict[eor[0]] = image

    with pytest.raises(KeyError):
        images.add_images(image_dict)

    # Now add other names
    image_dict = {}
    ext_ori_images = [
        [
            "INDIGO_2021-12-17_Z7II-B_0039_2",
            2643.0560,
            342715.0911648543551564,
            4.9515,
            -58.407,
            85.239,
            149.603,
        ],
        [
            "INDIGO_2021-12-17_Z7II-B_0040_2",
            2643.2077,
            342717.0722784474492073,
            4.9169,
            -53.782,
            86.843,
            144.631,
        ],
    ]

    for eor in ext_ori_images:
        position = np.array([eor[1], eor[2], eor[3]])
        orientation = Rotation.from_opk_degree(eor[4], eor[5], eor[6])
        image = ImagePerspective(
            width=8256,
            height=5504,
            position=position,
            orientation=orientation,
            crs=None,
            camera=cam_20mm,
        )
        image_dict[eor[0]] = image

    images.add_images(image_dict)
    assert len(images) == 12


def test___setter(images):

    with pytest.raises(ValueError):
        images["new_image"] = "string is not an image"

    images["new_image"] = images.index(0)
    assert len(images) == 11


def test___getter(images):

    res = images[["INDIGO_2021-12-17_Z7II-B_0039", "INDIGO_2021-12-17_Z7II-B_0040"]]
    assert len(res) == 2

    with pytest.raises(KeyError):
        images["wrong_image_key"]


def test_index(images):
    with pytest.raises(IndexError):
        images.index(100)

    res = images.index([1, 3])
    assert len(res) == 2


def test_empty_image_batch_exceptions():
    images = ImageBatch(images={})

    with pytest.raises(ValueError):
        images.map_center_point()

    with pytest.raises(ValueError):
        images.map_footprint()

    with pytest.raises(ValueError):
        images.project(np.array([[0, 0, 0]]))


def test_map_center_point(images):
    result = images.map_center_point()

    assert len(result) == 10

    res_image = result["INDIGO_2021-12-17_Z7II-B_0039"]
    assert res_image.ok is True
    assert np.isclose(res_image.gsd, 0.001, atol=0.0001, rtol=0.0)
    assert np.allclose(
        res_image.coordinates,
        np.array([2.63808856e03, 3.42714739e05, 4.73475821e00]),
        atol=0.001,
        rtol=0.0,
    )

    # We will add 3 images
    # the new images will point away from the mesh
    # but the first one has no camera model
    # the second one has no mapper specified
    # the third should be returning false because of wrong direction
    image_dict = {}
    ext_ori_images = [
        [
            "fake_1",
            2643.0560,
            342715.0911648543551564,
            4.9515,
            -58.407,
            -85.239,
            149.603,
            None,
            None,
        ],
        [
            "fake_2",
            2643.2077,
            342717.0722784474492073,
            4.9169,
            -53.782,
            -86.843,
            144.631,
            images.index(0).camera,
            None,
        ],
        [
            "fake_3",
            2643.2077,
            342717.0722784474492073,
            4.9169,
            -53.782,
            -86.843,
            144.631,
            images.index(0).camera,
            images.index(0).mapper,
        ],
    ]

    for eor in ext_ori_images:
        position = np.array([eor[1], eor[2], eor[3]])
        orientation = Rotation.from_opk_degree(eor[4], eor[5], eor[6])
        image = ImagePerspective(
            width=8256,
            height=5504,
            position=position,
            orientation=orientation,
            mapper=eor[8],
            camera=eor[7],
        )
        image_dict[eor[0]] = image

    images.add_images(image_dict)

    result = images.map_center_point()

    assert len(result) == 13
    assert result["fake_1"].ok is False
    assert Issue.IMAGE_BATCH_ERROR in result["fake_1"].issues
    assert Issue.IMAGE_BATCH_ERROR in result["fake_1"].issues
    assert result["fake_2"].ok is False
    assert result["fake_3"].ok is False
    assert Issue.NO_INTERSECTION in result["fake_3"].issues

    images_2 = ImageBatch({"a": images.index(0)})
    # Plane with altitude over all other
    # None of the center points can be mapped
    mapper = MappingHorizontalPlane(8000, crs=None)
    result = images_2.map_center_point(mapper=mapper)
    for _, res in result.items():
        assert res.ok is False


def test_map_footprint(images):
    # We change the position to be closer to the wall
    # just because the mesh is very tigh cut on the edges. And the image was part of the mesh creation
    # So for some areas where the image may have no overlap there would be no mesh
    images["INDIGO_2021-12-17_Z7II-B_0044"]._position = np.array([2638.7455, 342718.9792082067579031, 5.0487])

    result = images.map_footprint()

    assert len(result) == 10

    res_image = result["INDIGO_2021-12-17_Z7II-B_0044"]
    assert res_image.ok is True
    assert np.isclose(res_image.gsd, 0.00025, atol=0.0001, rtol=0.0)
    assert np.allclose(
        res_image.coordinates[0],
        np.array([2.63788572e03, 3.42718522e05, 4.25969046e00]),
        atol=0.001,
        rtol=0.0,
    )

    # Plane with altitude over all other
    mapper = MappingHorizontalPlane(8000, crs=None)

    map_result = images.map_footprint(mapper=mapper)
    for _, res in map_result.items():
        assert res.ok is False


def test_project(images):
    projection_dict = images.project(np.array([2637.874263, 342718.336006, 6.225628]))
    assert projection_dict is not None
    assert len(projection_dict) == 10

    projection_dict = images.project(np.array([2637.874263, 342718.336006, 6.225628]), only_valid=True)
    assert projection_dict is not None
    assert len(projection_dict) == 8

    # No valid projections
    projection_dict = images.project(np.array([2637.874263, 342718.336006, 80000]), only_valid=True)
    assert projection_dict is None

    # Fake that all images have no valid georeference
    for _, v in images.images.items():
        v._position = None
    # No valid projections
    projection_dict = images.project(np.array([2637.874263, 342718.336006, 80000]), only_valid=True)
    assert projection_dict is None
