.. _mapper_georef_array:

=====================
GeorefArray Mapper
=====================

The georef-array mapper :py:class:`weitsicht.MappingGeorefArray` is an in-memory mapping backend based on:

- a 2D numpy array of height values (rows × cols), and
- an affine geo-transform that maps pixel indices to coordinates in the mapper CRS.

It is commonly used as a cached backend for repeated operations on a limited raster extent (see
:py:meth:`weitsicht.MappingRaster.load_window`) with advanced ray-bilinear intersection operation.

.. important::
   You can retrieve a :py:class:`~weitsicht.MappingGeorefArray` from :py:class:`~weitsicht.MappingRaster` via
   ``preload_window`` / ``preload_full_raster`` or :py:meth:`~weitsicht.MappingRaster.load_window`.

When to use this mapper
=======================

Use :py:class:`~weitsicht.MappingGeorefArray` when:

- you already have a height grid in memory, or
- you use :py:class:`~weitsicht.MappingRaster` but want to cache a raster window in memory and avoid further disk IO.
- higher accuracy: GeorefArray mapper is providing accuarate ray-bilinear batch intersection.

Compared to :py:class:`~weitsicht.MappingRaster`, this backend is also useful when you want a more accurate ray/surface
intersection model within each raster cell (see below).

CRS and projections
===================

The geo-transform defines how pixel indices map to mapper CRS coordinates, so the mapper CRS can be any CRS supported by
pyproj/PROJ (e.g. UTM, Lambert, local projected systems). It can also be a geographic (lon/lat) CRS as long as the
geo-transform and sampling assumptions are meaningful for your area of interest.

As with other mappers, mapping methods accept either ``crs_s=...`` (source CRS) or a pre-built
``transformer=...`` to transform input rays/coordinates into the mapper CRS.

.. important::
   Ray traversal in this backend samples rays in 50 m steps up to 8 km.
   This assumes the ray input CRS is eucledian coordinate system (e.g. cratesian crs) with units meters (or another linear unit where “50” and “8000” represent real distances).

   If your rays or ray orientation is given in lon/lat degrees, transform poses/rays to a projected (meter-based) CRS before
   using ``map_coordinates_from_rays``.

How ray intersections work (bilinear patch intersection)
=========================================================

Many terrain intersection workflows approximate a raster DEM/DSM as a height function ``z = f(x, y)`` and try to find a
ray parameter ``t`` such that::

   ray_z(t) = f(ray_x(t), ray_y(t))

If ``f`` is modeled as a bilinear interpolation surface within each raster cell, then each cell forms a *bilinear patch*
defined by its four corner heights. Intersecting a ray with that surface is not a single plane intersection; it requires
solving for the intersection with the (generally) non-planar bilinear patch.

:py:class:`~weitsicht.MappingGeorefArray` uses a bilinear patch intersection approach:

1. Identify candidate raster cells along a ray segment.
2. For each candidate cell, build the 3D bilinear patch from the four corner points.
3. Compute the ray–patch intersection within that cell.

Implementation outline (ray sampling + raster traversal)
--------------------------------------------------------

To make CRS handling more robust (and to avoid transforming a direction vector through potentially non-linear CRS
operations), the current implementation samples the ray into short segments and works in *pixel space* for candidate cell
selection:

1. Split the ray into 50 m segments up to 8 km (currently fixed; assumes the ray CRS uses meters).
2. Transform the sampled 3D points into the mapper CRS.
3. Convert these points into raster pixel coordinates using the affine geo-transform.
4. For each 2D segment in pixel space, enumerate all raster cells the segment traverses (see figure below), using
   :py:func:`weitsicht.geometry.line_grid_intersection.line_grid_intersection_points`.
5. For every traversed cell, test the 3D ray segment against the cell’s bilinear patch in shifted pixel coordinates
   (internally a ``-0.5`` shift is used so raster samples lie on integer pixel nodes).

.. figure:: /_static/raster_traversal.svg
  :align: center
  :width: 360
  :alt: Raster traversal in 2D pixel space

  Raster traversal in 2D pixel space to enumerate candidate raster cells touched by a ray segment.

.. note:: Per ray: Finding the traversed segments and checking upfront possible cells is done via numpy matrix operations and some effort was put to speed that up.
   Only over valid candidates will be itterated to check the bilinear path for intersection. This speeds up the approach significant rather than itteratively traversing cell by cell (like in DDA algorithm).

Why this can be more accurate
-----------------------------

An iterative ray/plane approach (as used by :py:class:`~weitsicht.MappingRaster` for ray intersections) repeatedly:

- intersects the ray with a horizontal plane at a current height guess,
- samples a height at the resulting (x, y),
- updates the height guess and iterates until convergence.

That iteration can be a good approximation, but it depends on convergence behavior and on how heights are sampled (point
sampling vs. bilinear interpolation). A bilinear patch intersection models the cell surface continuously and computes the
intersection within the cell directly, which can reduce bias and improve consistency for steep rays or quickly varying
terrain. AS well MappingRaster could miss intersections along the ray which are detected by bilinear patch intersections.

Performance and memory
=======================

- **No disk IO once loaded**: this mapper operates purely on the provided in-memory array.
- **Memory use**: proportional to the loaded window size (full raster if you cache everything).
- **Loading into memory**: Loading a full raster of severell GB can take a significant amount of time and even fail if your machine does not provide sufficient memory.

Minimal example (cache a raster window)
=======================================

.. code-block:: python

   from pyproj import CRS
   from weitsicht import MappingRaster

   mapper = MappingRaster(raster_path="dem.tif", crs=CRS.from_epsg(32633))

   # Cache a window (xmin, ymin, xmax, ymax) into memory (returns the MappingGeorefArray)
   georef_mapper = mapper.load_window((600_000, 5_340_000, 602_000, 5_342_000))

   # After caching, mapper.map_* automatically uses the in-memory backend:
   # hits = mapper.map_coordinates_from_rays(ray_vectors, ray_starts, crs_s=image_crs)

   # You can also call the cached backend explicitly:
   # hits = georef_mapper.map_coordinates_from_rays(ray_vectors, ray_starts, crs_s=image_crs)

Related pages
=============

- :doc:`raster` (disk-backed raster access with optional window caching)
- :doc:`horizontal_plane` (fast flat approximation)
- :doc:`mesh` (triangle mesh intersections)
- :doc:`../transformation` (CRS / vertical transformation notes)
