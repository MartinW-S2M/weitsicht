:py:class:`~weitsicht.transform.wgs84_local_tangent.WGS84LocalTangent`
======================================================================

Helpers for a WGS84 local tangent frame (ENU/NED) and conversions to/from WGS84 ECEF.

.. currentmodule:: weitsicht.transform.wgs84_local_tangent

.. rubric:: Class Methods

.. autosummary::

   WGS84LocalTangent.from_wgs84ell_crs
   WGS84LocalTangent.from_wgs84ell_orthometric
   WGS84LocalTangent.from_crs

.. rubric:: Methods

.. autosummary::

   WGS84LocalTangent.ell_to_ecef
   WGS84LocalTangent.vector_to_ecef
   WGS84LocalTangent.vector_from_ecef
   WGS84LocalTangent.point_to_ecef
   WGS84LocalTangent.point_from_ecef
   WGS84LocalTangent.to_ecef_matrix
   WGS84LocalTangent.to_ltp_matrix

.. rubric:: Properties

.. autosummary::

   WGS84LocalTangent.origin_ecef
   WGS84LocalTangent.r_ned_to_ecef
   WGS84LocalTangent.r_ecef_to_enu
   WGS84LocalTangent.r_enu_to_ecef

.. autoclass:: WGS84LocalTangent
   :show-inheritance:
   :special-members: __init__
