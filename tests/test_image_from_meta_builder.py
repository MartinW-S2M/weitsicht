import json
from pathlib import Path

import numpy as np
from pyproj import CRS

import weitsicht
from weitsicht import ImageFromMetaBuilder, PyExifToolTags, eor_from_meta, image_from_meta

DATA_DIR = Path(__file__).parent.resolve() / "data"
DJI_P1_TAGS_JSON = DATA_DIR / "image_exiftool_tags.json"


def _load_meta() -> dict:
    with DJI_P1_TAGS_JSON.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_eor_from_meta_zenmuse_p1():
    weitsicht.allow_ballpark_transformations()
    weitsicht.allow_non_best_transformations()

    tags = PyExifToolTags(_load_meta())
    result = eor_from_meta(tags.get_all())

    assert result.ok is True

    assert result.crs is not None
    assert result.crs.to_wkt() == CRS("EPSG:32636+3855").to_wkt()

    assert result.position.shape == (3,)
    assert np.all(np.isfinite(result.position))
    assert 0.0 < float(result.position[0]) < 1_000_000.0  # easting
    assert 0.0 < float(result.position[1]) < 10_000_000.0  # northing

    rot = result.orientation.matrix
    assert rot.shape == (3, 3)
    assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-6)
    assert abs(float(np.linalg.det(rot)) - 1.0) < 1e-6


def test_image_from_meta_builder():
    weitsicht.allow_ballpark_transformations()
    weitsicht.allow_non_best_transformations()

    tags = PyExifToolTags(_load_meta())

    builder = ImageFromMetaBuilder(tags)
    ior_1 = builder.ior()
    ior_2 = builder.ior()
    assert ior_1 is ior_2
    assert ior_1.ok is True
    assert (ior_1.width, ior_1.height) == (8192, 5460)

    eor_1 = builder.eor()
    eor_2 = builder.eor()
    assert eor_1 is eor_2
    assert eor_1.ok is True

    img_builder = builder.image()
    assert img_builder.ok is True
    assert img_builder.image.is_geo_referenced is True
    assert (img_builder.image.width, img_builder.image.height) == (8192, 5460)
    assert img_builder.image.crs is not None

    img_func = image_from_meta(tags)
    assert img_func.ok is True
    assert img_func.image.is_geo_referenced is True
    assert (img_func.image.width, img_func.image.height) == (8192, 5460)
    assert img_func.image.crs is not None
    assert img_func.image.crs.to_wkt() == img_builder.image.crs.to_wkt()
