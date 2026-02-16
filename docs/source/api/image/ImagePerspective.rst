
:py:class:`ImagePerspective`
================================

.. currentmodule:: weitsicht

.. rubric:: Main API Methods

.. autosummary::
  ImagePerspective.pixel_to_ray_vector
  ImagePerspective.project
  ImagePerspective.map_center_point
  ImagePerspective.map_footprint
  ImagePerspective.map_points


.. rubric:: additional helpful Methods

.. autosummary::

  ImagePerspective.from_dict
  ImagePerspective.position_to_crs

.. rubric:: Properties

.. autosummary::

  ImagePerspective.type
  ImagePerspective.param_dict

  ImagePerspective.mapper
  ImagePerspective.is_geo_referenced
  ImagePerspective.position_wgs84
  ImagePerspective.position_wgs84_geojson
  ImagePerspective.crs
  ImagePerspective.crs_wkt
  ImagePerspective.crs_proj4
  ImagePerspective.width
  ImagePerspective.height
  ImagePerspective.shape


.. autoclass:: ImagePerspective
   :special-members: __init__
   :show-inheritance:
