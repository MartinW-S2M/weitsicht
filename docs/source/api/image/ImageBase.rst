
:py:class:`ImageBase`
================================

.. currentmodule:: weitsicht

.. rubric:: Main API Methods

This are most important functions to be used for image classes

.. autosummary::

  ImageBase.project
  ImageBase.map_center_point
  ImageBase.map_footprint
  ImageBase.map_points


.. rubric:: additional helpful Methods

.. autosummary::

  ImageBase.from_dict
  ImageBase.position_to_crs

.. rubric:: Properties

.. autosummary::

  ImageBase.type
  ImageBase.param_dict

  ImageBase.mapper
  ImageBase.is_geo_referenced
  ImageBase.position_wgs84
  ImageBase.position_wgs84_geojson
  ImageBase.crs
  ImageBase.crs_wkt
  ImageBase.crs_proj4
  ImageBase.width
  ImageBase.height
  ImageBase.shape


.. autoclass:: ImageBase
   :special-members: __init__
