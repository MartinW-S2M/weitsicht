from weitsicht.mapping.base_class import MappingBase


def test_crs():
    mapper = MappingBase()

    assert mapper.crs is None


def test_crs_wkt():
    mapper = MappingBase()
    assert mapper.crs_wkt is None
