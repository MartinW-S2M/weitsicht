============
Top-Tips
============

PyProj
========
If special transformation grids are included in the coordinate transformation use:

.. code-block:: python

  # PROJ activate network grid/data
  pyproj.network.set_network_enabled(True)  # pyright: ignore[reportPrivateImportUsage]

Or specify data-dir via

.. code-block:: python

  pyproj.datadir.set_data_dir(proj_data_dir)


Results
========
The most important functions, mapping(``mapper.map_points``, ``mapper.map_footprint``, ``mapper.map_center_point``)
and coordinate projections (``image.project``) return ``MappingResult`` and ``ProjectionResult``

- ``MappingResult`` is either ``ResultFailure`` or ``MappingResultSuccess``
- ``ProjectionResult`` is either ``ResultFailure`` or ``ProjectionResultSuccess``

You can test whether it's Success or Failure be checking ``.ok``.
**The results will always be in the same order as the input pixel/coordinates.**

Always check for issues to catch reasons for invalid results

Errors
======
# TODO Check for Errors


Perspective Images
==================

.. tip::
   For drone photos, prefer building your ``ImagePerspective`` from EXIF/XMP metadata (e.g. ExifTool -> ``PyExifToolTags``
   -> ``image_from_meta()``) or use ``ImageFromMetaBuilder`` if you want IOR/EOR separately. See :doc:`metadata` and
   :ref:`example-0401`.

CRS quick reference
-------------------

.. list-table:: CRS overview (rows) vs. weitsicht components (columns)
   :header-rows: 1
   :stub-columns: 1

   * - CRS / units
     - Perspective Image (EOR)
     - Orthophoto CRS
     - Mapper Horizontal
     - MapperRaster GeorefArray
     - Mapper Trimesh
   * - ``None`` (local coordinates, no transforms)
     - OK
     - OK
     - OK
     - OK (``force_no_crs=True``)
     - OK
   * - Projected/local Cartesian **(0)**
     - OK
     - OK
     - OK
     - OK
     - OK
   * - Geocentric Cartesian (ECEF) (e.g. ``EPSG:4978``)
     - OK
     - Not possible - Rare (never seen such thing)
     - Not possible at the moment **(1)**
     - Not possible - Rare (never seen such thing)
     - OK
   * - Geodetic lon/lat (degrees) (e.g. ``EPSG:4326`` / ``EPSG:4979``)
     - Not working **(2)**
     - Possible (but units for gsd/area are degrees)
     - Possible **(3)**
     - Possible (only with 3D CRS) **(4)**
     - Not working (would need transformation first)

| **(0)** has to be 3D CRS or compound, meters (e.g. UTM / ``EPSG:25833+3855``)
| **(1)** ECEF and plane working mathematically but plane normal 0,0,1 makes no sense here. Will implement nonHorizontal plane soon.
| **(2)** use as input CRS and transform to a metric CRS first, also orientation needs to be transformed
| **(3)** using this a lot for sea-side EPSG:4326+3855 -> WGS84 (Lat/Lon) + EGM2008
| **(4)** beware units/vertical datum, very large raster resolutions may lead to problems


Mapping
============
For raster-based mapping you can trade disk IO for memory:

- ``MappingRaster`` (default) keeps the raster on disk and reads only the needed values.
- If you cache a window/full raster (``preload_window``, ``preload_full_raster`` or ``load_window``),
  :class:`~weitsicht.MappingRaster` automatically uses an in-memory :class:`~weitsicht.MappingGeorefArray` backend.

.. code-block:: python

  from pyproj import CRS
  from weitsicht import MappingRaster

  mapper = MappingRaster(
      raster_path="dem.tif",
      crs=CRS.from_epsg(32633),
      preload_window=(600_000, 5_340_000, 602_000, 5_342_000),
  )

  # mapper.map_* now uses the cached backend
  res = mapper.map_heights_from_coordinates([[601_000, 5_341_000]], crs_s=mapper.crs)

  # Or cache later (returns MappingGeorefArray)
  georef = mapper.load_window((600_000, 5_340_000, 602_000, 5_342_000))

.. important::
  If you cache only a window, mapping is limited to that cached extent (outside points/rays will report
  ``Issue.OUTSIDE_RASTER``).
