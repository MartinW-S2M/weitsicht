from pathlib import Path

import pytest
from pyproj import CRS

from weitsicht.mapping.base_class import MappingType
from weitsicht.mapping.mapping_dict_selector import get_mapper_from_dict

DATA_DIR = Path(__file__).parent.resolve() / "data"


def test_mapper_load_from_dict():
    # wrong dictionary, no Type present
    dict_fail = {"raster_filepath": "wrongfile", "crs": CRS("EPSG:31287+5778").to_wkt()}

    with pytest.raises(KeyError):
        get_mapper_from_dict(dict_fail)

    # wrong type
    dict_fail = {
        "type": "PointCloud",
        "raster_filepath": "wrongfile",
        "crs": CRS("EPSG:31287+5778").to_wkt(),
    }

    with pytest.raises(ValueError):
        get_mapper_from_dict(dict_fail)

    # Horizontal dict working
    dict_horizontal = {
        "type": "horizontalPlane",
        "plane_altitude": 43.0,
        "crs": "EPSG:4979",
    }
    mapper = get_mapper_from_dict(dict_horizontal)
    assert mapper.type.fullname is MappingType.HorizontalPlane.fullname

    # Raster dict working
    dict_raster = {
        "type": "Raster",
        "raster_filepath": (DATA_DIR / "dhm_at_lamb_10m_2018.tif").as_posix(),
        "crs": CRS("EPSG:31287+5778").to_wkt(),
    }

    mapper = get_mapper_from_dict(dict_raster)
    assert mapper.type.fullname is MappingType.Raster.fullname

    # Raster Fail
    dict_raster = {
        "type": "Raster",
        "raster_filepath": "wrongfile",
        "crs": CRS("EPSG:31287+5778").to_wkt(),
    }

    with pytest.raises(FileNotFoundError):
        get_mapper_from_dict(dict_raster)

    # Mesh working dict
    mesh_path = DATA_DIR / "decimated_160k_31256.ply"
    dict_mesh = {
        "type": MappingType.Trimesh.fullname,
        "mesh_filepath": mesh_path.as_posix(),
        "crs": None,
    }
    mapper = get_mapper_from_dict(dict_mesh)
    assert mapper.type.fullname == MappingType.Trimesh.fullname

    # Mesh dict not working
    mesh_path = DATA_DIR / "decimated_160k_31256_wrong.ply"
    dict_mesh = {
        "type": MappingType.Trimesh.fullname,
        "mesh_filepath": mesh_path.as_posix(),
        "crs": None,
    }
    with pytest.raises(FileNotFoundError):
        get_mapper_from_dict(dict_mesh)
