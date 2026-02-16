
:py:class:`MappingGeorefArray`
==============================

For deeper information about the concept see: :ref:`mapper_georef_array`

.. currentmodule:: weitsicht

.. rubric:: Class Methods

.. autosummary::

  MappingGeorefArray.from_dict

.. rubric:: Main API Methods

.. autosummary::

  MappingGeorefArray.map_coordinates_from_rays
  MappingGeorefArray.map_heights_from_coordinates


.. rubric:: Methods Raster specific

This functions can be used to interact with the raster and coordinates

.. autosummary::

  MappingGeorefArray.pixel_to_coordinate
  MappingGeorefArray.pixel_valid
  MappingGeorefArray.coordinate_on_raster
  MappingGeorefArray.coordinate_to_pixel


.. rubric:: Properties

.. autosummary::

  MappingGeorefArray.type
  MappingGeorefArray.param_dict
  MappingGeorefArray.crs
  MappingGeorefArray.crs_wkt
  MappingGeorefArray.transform
  MappingGeorefArray.width
  MappingGeorefArray.height


.. autoclass:: MappingGeorefArray
   :special-members: __init__
   :show-inheritance:
