=======
Mappers
=======

Mappers are the ground / surface model used by images and batches to intersect rays or sample heights.
They work in their own CRS and expose a small, consistent API.

Core interfaces
---------------

- ``map_coordinates_from_rays(ray_vectors, ray_starts, crs_s=None, transformer=None)``:
  intersects rays with the surface and returns a ``MappingResult`` (coordinates + mask + issues).
- ``map_heights_from_coordinates(coordinates, crs_s=None, transformer=None)``:
  returns 3D coordinates with heights injected/interpolated from the surface model.
- ``param_dict`` / ``from_dict``:
  serialize and reconstruct mapper configuration.
- ``crs`` / ``crs_wkt``:
  mapper CRS; inputs are transformed automatically when ``crs_s`` differs.

Available mappers
-----------------

- ``MappingHorizontalPlane``: constant-altitude plane; fastest for flat scenes.
- ``MappingRaster``: samples a raster DEM/DSM (rasterio-based); good for terrain.
- ``MappingGeorefArray``: in-memory height grid (often produced via ``MappingRaster.load_window(...)``).
- ``MappingTrimesh``: ray intersections with a 3D mesh via ``trimesh``; supports complex surfaces.

When to choose which
--------------------

- Flat site / sea surface / quick tests → ``MappingHorizontalPlane`` with a known altitude.
- Terrain from DEM/DSM → ``MappingRaster`` (disk-backed; bilinear height sampling).
- In-memory raster subset → ``MappingGeorefArray`` (fast repeated operations, no disk IO once loaded).
- Detailed surface mesh → ``MappingTrimesh`` (ray–triangle hits, optional coordinate shift).

Using the core methods
----------------------

.. code-block:: python

   # Rays (e.g. from ImagePerspective.pixel_to_ray_vector) to ground
   result = mapper.map_coordinates_from_rays(ray_vecs, ray_starts, crs_s=image.crs)
   if result.ok:
       ground_pts = result.coordinates[result.mask]
   else:
       print("Failed:", result.error, "Issues:", result.issues)

   # Heights at XY positions
   result_h = mapper.map_heights_from_coordinates(xy, crs_s=target_crs)
   if result_h.ok:
       pts_with_z = result_h.coordinates[result_h.mask]

Notes & tips
------------

- Align mapper CRS with your images when possible; otherwise always pass ``crs_s`` (or reuse a ``transformer``).
- Expect partial success: check ``result.ok`` and filter with ``result.mask``; inspect ``issues`` for warnings like
  ``WRONG_DIRECTION``, ``OUTSIDE_RASTER``, ``RASTER_NO_DATA``, ``MAX_ITTERATION``, or ``NO_INTERSECTION``.
- Heavy mappers (rasters/meshes) typically open external resources at construction—reuse them across many images/batches.

  - For :class:`~weitsicht.MappingRaster`, raster data is read lazily during runtime operations by default; it is only
    loaded into memory when you enable ``preload_full_raster`` / ``preload_window`` or call
    :meth:`~weitsicht.MappingRaster.load_window`.

Raster backend selection (``MappingRaster``)
--------------------------------------------

``MappingRaster`` has two internal backends:

- **disk-backed** (default): samples/intersects directly from the raster file via ``rasterio``
- **in-memory** (optional): a cached :class:`~weitsicht.MappingGeorefArray` stored in ``mapper.georef_array``

When an in-memory backend is present, ``MappingRaster.map_coordinates_from_rays`` and
``MappingRaster.map_heights_from_coordinates`` automatically delegate to it.

Load a cached window / full raster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pyproj import CRS
   from weitsicht import MappingRaster

   mapper = MappingRaster(
       raster_path="dem.tif",
       crs=CRS.from_epsg(32633),
       preload_window=(600_000, 5_340_000, 602_000, 5_342_000),
   )

   # Uses MappingGeorefArray because a window is cached
   res = mapper.map_heights_from_coordinates([[601_000, 5_341_000]], crs_s=mapper.crs)

   # You can also cache after construction (returns the MappingGeorefArray)
   georef = mapper.load_window((600_000, 5_340_000, 602_000, 5_342_000))
   assert georef is mapper.georef_array

   # Optional helpers
   backend = mapper.backend          # MappingGeorefArray if loaded, else MappingRaster
   georef2 = mapper.georef_mapper    # raises if not loaded

.. important::
   If you cache only a window, mapping operations are limited to that window extent. Points/rays outside the cached
   extent will return ``Issue.OUTSIDE_RASTER``.
