import numpy as np
import pyproj.network
from pyproj import CRS, Transformer

from weitsicht.transform.coordinates_transformer import CoordinateTransformer

pyproj.network.set_network_enabled(True)  # pyright: ignore[reportPrivateImportUsage]


def test_from_pipeline():
    # Need to do that. Need proper pipeline for testing
    assert True


def test_transform():
    # MGI Lambert in Bessel Ellipsoid
    ref_p_1_mgi_lambert_ell = np.array([556538.9876, 429158.5652, 2000])
    # from BEV transformator https://transformator.bev.gv.at/at.gv.bev.transformator/austrian
    # ref_p_1_mgi_9266 = np.array([4142953.6487, 1142861.1838, 4698745.5662])
    ref_p1_mgi_9267_bessel_ell = np.array([15.42187111, 47.743444717, 2000.000])

    # lambert with ellipsoid (bessel) height to MGI
    # Directly using pyproj Transformer
    t_0_1 = Transformer.from_crs(CRS(31287).to_3d(), 9267, always_xy=True, allow_ballpark=False, only_best=True)
    p_0_1 = np.array(
        t_0_1.transform(
            ref_p_1_mgi_lambert_ell[0],
            ref_p_1_mgi_lambert_ell[1],
            ref_p_1_mgi_lambert_ell[2],
        )
    )
    assert np.allclose(p_0_1, ref_p1_mgi_9267_bessel_ell, atol=0.001, rtol=0)
    # Using CoordinateTransformer class
    c_0_1 = CoordinateTransformer.from_crs(CRS(31287).to_3d(), CRS(9267))
    assert c_0_1 is not None
    p_0_1_c = c_0_1.transform(ref_p_1_mgi_lambert_ell)
    assert np.allclose(p_0_1_c, ref_p1_mgi_9267_bessel_ell, atol=1e-8, rtol=0)
    # Backwards
    p_0_1_c_inverse = c_0_1.transform(p_0_1_c, direction="inverse")
    assert p_0_1_c_inverse is not None
    assert np.allclose(p_0_1_c_inverse, ref_p_1_mgi_lambert_ell, atol=0.001, rtol=0)

    # USING national height system (grids involved)
    # MGI Lambert in gha
    ref_p_1_mgi_lambert_gha = np.array([556538.9876, 429158.5652, 2000])
    # from BEV transformator https://transformator.bev.gv.at/at.gv.bev.transformator/austrian
    ref_p1_mgi_4312_gha = np.array([15.421871111, 47.743444716, 2000.000])

    # lambert with gha heights to MGI (lat,lon) gha
    crs_lambert_mgi_gha = CRS("EPSG:31287+9274")
    # 9267 is geog3d with ell height. We need to combine 2d ell with gha
    crs_mgi_latlon_gha = CRS("EPSG:4312+9274")

    # using pyproj Transformer
    t_0_2 = Transformer.from_crs(
        crs_lambert_mgi_gha,
        crs_mgi_latlon_gha,
        always_xy=True,
        allow_ballpark=False,
        only_best=True,
    )

    p_0_2 = t_0_2.transform(
        ref_p_1_mgi_lambert_gha[0],
        ref_p_1_mgi_lambert_gha[1],
        ref_p_1_mgi_lambert_gha[2],
    )
    assert np.allclose(p_0_2, ref_p1_mgi_4312_gha, atol=1e-9, rtol=0)

    # using CoordinateTransformer class
    c_0_2 = CoordinateTransformer.from_crs(crs_lambert_mgi_gha, crs_mgi_latlon_gha)
    assert c_0_2 is not None
    p_0_2_c = c_0_2.transform(ref_p_1_mgi_lambert_gha)
    assert np.allclose(p_0_2_c, ref_p1_mgi_4312_gha, atol=1e-8, rtol=0)

    # backwards
    assert c_0_2 is not None
    p_0_2_c_inverse = c_0_2.transform(ref_p1_mgi_4312_gha, direction="inverse")
    assert np.allclose(p_0_2_c_inverse, ref_p_1_mgi_lambert_gha, atol=1e-3, rtol=0)

    # (2) test to MGI GK and to UTM with different heights
    # Most references are taken from BEV transformator
    # BEV transformator https://transformator.bev.gv.at/at.gv.bev.transformator/austrian
    # If using pyproj it will get all transformations and epsg from a database of the current pyproj version
    # If nothing specified it will use the best transformation also containing v and h grids.

    # Using heights in use of Austria GHA (Gebrauchshoehen)
    # Lambert to GK MGI
    # That one works well
    pt_test_lambert = np.array([556538.9876, 429158.5652, 2000])
    # From BEV transformator
    ref_mgi_gk_31_gha = np.array([156606.740, 5291475.650, 2000.000])

    crs_lambert_mgi_gha = CRS("EPSG:31287+9274")
    crs_gk_31_gha = CRS("EPSG:9272+9274")
    c_1_1 = CoordinateTransformer.from_crs(crs_lambert_mgi_gha, crs_gk_31_gha)
    assert c_1_1 is not None
    p_1_1_c = c_1_1.transform(pt_test_lambert)
    assert np.allclose(ref_mgi_gk_31_gha, p_1_1_c, atol=0.001, rtol=0)
    # Backwards
    p_transformed_inverse = c_1_1.transform(ref_mgi_gk_31_gha, direction="inverse")
    assert np.allclose(pt_test_lambert, p_transformed_inverse, atol=0.001, rtol=0)

    # Using heights in use of Austria GHA (Gebrauchshoehen) for Lambert to UTM with ell heights
    # That one works well
    pt_test_lambert = np.array([556538.9876, 429158.5652, 2000])
    # From BEV transformator -  UTM ell heights grs80
    pt_ref_utm_ell = np.array([531547.645, 5287818.137, 2047.163])

    crs_lambert_mgi_gha = CRS("EPSG:31287+5778")
    crs_etrs_utm_ell = CRS("EPSG:25833").to_3d()  # We need to make utm 3d that its using 3d transformation correctly

    c_2_2 = CoordinateTransformer.from_crs(crs_lambert_mgi_gha, crs_etrs_utm_ell)
    assert c_2_2 is not None
    p_2_2_c = c_2_2.transform(pt_test_lambert)
    assert np.allclose(p_2_2_c, pt_ref_utm_ell, atol=0.001, rtol=0)

    # Backwards
    p_2_2_c_inverse = c_2_2.transform(pt_ref_utm_ell, direction="inverse")
    assert np.allclose(pt_test_lambert, p_2_2_c_inverse, atol=0.001, rtol=0)

    # Using heights in use of Austria GHA (Gebrauchshoehen) for Lambert to UTM with egm2008 heights
    # That one works well
    pt_test_lambert = np.array([556538.9876, 429158.5652, 2000])

    # From BEV transformator
    pt_ref_utm_ell = np.array([531547.645, 5287818.137, 2047.163])
    # Self calculated directly from UTM ell to UTM egm2008
    # We need to make UTM to 3d that its aware of ellipsoid heights
    c_utm_ell_utm_egm2008 = Transformer.from_crs(CRS("EPSG:25833").to_3d(), "EPSG:25833+3855")
    pt_ref_utm_egm2008 = np.array(c_utm_ell_utm_egm2008.transform(*pt_ref_utm_ell))

    crs_lambert_mgi_gha = CRS("EPSG:31287+5778")
    crs_etrs_utm_egm2008 = CRS("EPSG:25833+3855")
    c_2_3 = CoordinateTransformer.from_crs(crs_lambert_mgi_gha, crs_etrs_utm_egm2008)
    assert c_2_3 is not None
    p_2_3_c = c_2_3.transform(pt_test_lambert)
    assert np.allclose(p_2_3_c, pt_ref_utm_egm2008, atol=0.001, rtol=0)

    # Backwards
    p_2_3_c_inverse = c_2_3.transform(pt_ref_utm_egm2008, direction="inverse")
    assert np.allclose(pt_test_lambert, p_2_3_c_inverse, atol=0.001, rtol=0)

    # Now we check that case why directly using pyproj Transformer can make a problem
    # This is where the direct use of pyproj Transformer fails

    p_wgs84_egm2008 = np.array([15.420831277, 47.742966114, 2000])
    # To get reference from BEV we will transform first to ell coordinates which is correct here anyhow
    t_3_0 = Transformer.from_crs(CRS("EPSG:4326+3855"), 4979, always_xy=True)
    p_wgs84_ell = t_3_0.transform(*p_wgs84_egm2008)

    print(p_wgs84_ell)  # That we use in BEV transformator
    p_ref_lambert_gha = np.array([556539.049, 429158.797, 2000.478])

    crs_wgs84_egm2008 = CRS("EPSG:4326+3855")
    crs_mgi_gha = CRS("EPSG:31287+5778")
    t_3_1 = Transformer.from_crs(
        crs_wgs84_egm2008,
        crs_mgi_gha,
        always_xy=True,
        allow_ballpark=False,
        only_best=True,
    )
    p_3_1 = t_3_1.transform(*p_wgs84_egm2008)
    # That point is now wrong
    print("wrong:", p_3_1, " diff:", p_3_1 - p_ref_lambert_gha)

    c_3_1 = CoordinateTransformer.from_crs(crs_wgs84_egm2008, crs_mgi_gha)
    assert c_3_1 is not None
    p_3_1_c = c_3_1.transform(p_wgs84_egm2008)
    # Using our Coordinate Transformer it works
    print("correct:", p_3_1_c, " diff:", p_3_1_c - p_ref_lambert_gha)

    # WGS84/UTM also does not work directly
    p_wgs84_utm_egm2008 = np.array([531547.712, 5287818.367, 2000.000])
    # To get reference from BEV we will transform first to ell coordinates which is correct here anyhow
    p_ref_lambert_gha = np.array([556539.049, 429158.797, 2000.478])

    crs_wgs84_utm_egm2008 = CRS("EPSG:32633+3855")
    crs_mgi_gha = CRS("EPSG:31287+5778")
    t_3_2 = Transformer.from_crs(
        crs_wgs84_utm_egm2008,
        crs_mgi_gha,
        always_xy=True,
        allow_ballpark=False,
        only_best=True,
    )
    p_3_2 = t_3_2.transform(*p_wgs84_utm_egm2008)
    # That point is now wrong
    print("wrong:", p_3_2, " diff:", p_3_2 - p_ref_lambert_gha)

    c_3_2 = CoordinateTransformer.from_crs(crs_wgs84_utm_egm2008, crs_mgi_gha)
    assert c_3_2 is not None
    p_3_2_c = c_3_2.transform(p_wgs84_utm_egm2008)
    # Using our Coordinate Transformer it works
    print("correct:", p_3_2_c, " diff:", p_3_2_c - p_ref_lambert_gha)
