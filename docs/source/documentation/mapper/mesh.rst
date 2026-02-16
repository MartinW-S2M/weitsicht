.. _mapper_mesh:

============
Mesh Mapper
============

.. note::
   Currently only the closest intersection is returned by ``map_coordinates_from_rays()``.

.. warning::
   Only cartesian-like coordinate systems (ECEF, UTM, ...) should be used for the mapper CRS.

The mesh mapper :py:class:`weitsicht.MappingTrimesh` wraps a triangular surface mesh and performs ray/triangle
intersection using `trimesh <https://trimesh.org/index.html>`__. It is designed for two common geospatial tasks:

- sampling surface heights for given (x, y) locations by casting vertical rays downwards
- intersecting camera (or LiDAR) rays with the mesh to obtain 3D hit points

Trimesh handles the heavy lifting via :mod:`trimesh.ray`, which builds a BVH (bounding volume hierarchy) to accelerate
intersection queries.

.. figure:: /_static/mesh_ray.jpg
   :width: 520
   :align: center
   :alt: Ray-mesh intersection

   Ray/mesh intersection used for height sampling and back-projection.

Transformation strategy
========================

The mesh itself is not reprojected. Instead, rays are transformed into the mesh CRS for intersection.

Implementation detail:
``weitsicht`` transforms multiple points along each ray (at fixed distances) and reconstructs a robust direction vector
in the target CRS. This avoids directly transforming a direction vector through potentially non-linear CRS operations.

Typical workflow
==================

1. Initialize the mapper from a mesh file path.
2. (Optional) specify ``crs`` and/or ``coordinate_shift`` if the mesh uses a local coordinate origin.
3. Use either:

   - ``map_heights_from_coordinates(...)`` for (x, y) → height sampling, or
   - ``map_coordinates_from_rays(...)`` for ray intersections.

Performance notes
=================

- For dense ray batches, trimesh may use accelerated backends when installed (platform-dependent).
- If the mesh has holes, rays can pass without intersections; these rays are returned as invalid.

Minimal example
===============

.. code-block:: python

   import numpy as np
   from pyproj import CRS
   from weitsicht import MappingTrimesh

   mapper = MappingTrimesh(mesh_path="terrain.ply", crs=CRS.from_epsg(32633))

   # Height lookup at ground coordinates
   xy = np.array([[10.0, 12.5], [15.0, 18.0]])
   res_h = mapper.map_heights_from_coordinates(xy, crs_s=mapper.crs)

   # Ray intersections (origins + vectors must be in the same source CRS)
   # hits = mapper.map_coordinates_from_rays(ray_vectors, ray_starts, crs_s=image_crs)

Related pages
=============

- :doc:`raster`
- :doc:`horizontal_plane`
- :doc:`../transformation`
