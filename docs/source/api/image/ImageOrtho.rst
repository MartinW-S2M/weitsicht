.. _ortho-image-class:


:py:class:`ImageOrtho`
======================

For deeper information about the concept see: :ref:`image_ortho`


.. currentmodule:: weitsicht

.. rubric:: Class Methods

.. autosummary::

  ImageOrtho.from_file
  ImageOrtho.from_dict

.. rubric:: Main API Methods

.. autosummary::

  ImageOrtho.project
  ImageOrtho.map_center_point
  ImageOrtho.map_footprint
  ImageOrtho.map_points


.. rubric:: additional helpful Methods

.. autosummary::

  ImageOrtho.from_dict
  ImageOrtho.position_to_crs

.. rubric:: Properties

.. autosummary::

  ImageOrtho.type
  ImageOrtho.param_dict

  ImageOrtho.mapper
  ImageOrtho.is_geo_referenced
  ImageOrtho.position_wgs84
  ImageOrtho.position_wgs84_geojson
  ImageOrtho.crs
  ImageOrtho.crs_wkt
  ImageOrtho.crs_proj4
  ImageOrtho.width
  ImageOrtho.height
  ImageOrtho.shape


.. autoclass:: ImageOrtho
   :special-members: __init__
   :show-inheritance:
