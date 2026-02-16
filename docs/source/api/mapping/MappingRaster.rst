
:py:class:`MappingRaster`
=========================

For deeper information about the concept see: :ref:`mapper_raster`

.. currentmodule:: weitsicht

.. rubric:: Class Methods

.. autosummary::

  MappingRaster.from_dict

.. rubric:: Main API Methods

.. autosummary::

  MappingRaster.map_coordinates_from_rays
  MappingRaster.map_heights_from_coordinates


.. rubric:: Methods Raster specific

This functions can be used to interact with the raster and coordinates

.. autosummary::

  MappingRaster.pixel_to_coordinate
  MappingRaster.pixel_valid
  MappingRaster.coordinate_on_raster
  MappingRaster.coordinate_to_pixel
  MappingRaster.get_coordinate_height
  MappingRaster.intersection_ray
  MappingRaster.load_window


.. rubric:: Properties

.. autosummary::

  MappingRaster.type
  MappingRaster.backend
  MappingRaster.georef_mapper
  MappingRaster.param_dict
  MappingRaster.crs
  MappingRaster.crs_wkt
  MappingRaster.resolution
  MappingRaster.transform
  MappingRaster.width
  MappingRaster.height


.. autoclass:: MappingRaster
   :special-members: __init__
   :show-inheritance:
