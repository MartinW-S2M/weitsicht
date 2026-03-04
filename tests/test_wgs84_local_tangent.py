import numpy as np
from pyproj import CRS

from weitsicht.transform import WGS84LocalTangent


def _rz(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def test_from_wgs84ell_elipsoid_origin_and_matrices():
    frame = WGS84LocalTangent.from_wgs84ell_elipsoid(lon_deg=0.0, lat_deg=0.0, h_m=0.0)

    assert frame.crs_ecef.to_epsg() == 4978
    assert frame._origin_ecef is not None
    assert frame._origin_ecef.shape == (3,)
    assert np.all(np.isfinite(frame._origin_ecef))

    # WGS84 semi-major axis at lon=0, lat=0, h=0.
    assert np.allclose(frame._origin_ecef, np.array([6378137.0, 0.0, 0.0]), atol=1e-6, rtol=0.0)

    r = frame.r_ecef_to_ned
    assert r.shape == (3, 3)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12, rtol=0.0)

    r_enu = frame.r_ecef_to_enu
    assert r_enu.shape == (3, 3)
    assert np.allclose(r_enu @ r_enu.T, np.eye(3), atol=1e-12, rtol=0.0)


def test_to_ecef_matrix_backforth_ned():
    frame = WGS84LocalTangent.from_wgs84ell_elipsoid(lon_deg=12.3, lat_deg=-45.6, h_m=123.0)

    r_local = _rz(np.deg2rad(30.0))
    r_ecef = frame.to_ecef_matrix(r_local, local_frame="NED")
    r_local_back = frame.to_ltp_matrix(r_ecef, local_frame="NED")

    assert np.allclose(r_local_back, r_local, atol=1e-12, rtol=0.0)


def test_from_crs_ecef_identity():
    frame_0 = WGS84LocalTangent.from_wgs84ell_elipsoid(lon_deg=0.0, lat_deg=0.0, h_m=10.0)
    assert frame_0._origin_ecef is not None
    x = float(frame_0._origin_ecef[0])
    y = float(frame_0._origin_ecef[1])
    z = float(frame_0._origin_ecef[2])

    frame_1 = WGS84LocalTangent.from_crs(x, y, z, CRS.from_epsg(4978))
    assert np.allclose(frame_1.origin_ecef, frame_0._origin_ecef, atol=1e-9, rtol=0.0)
    assert np.allclose(frame_1.r_ecef_to_ned, frame_0.r_ecef_to_ned, atol=1e-12, rtol=0.0)
