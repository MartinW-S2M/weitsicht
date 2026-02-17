.. _mapper_normals:

===============
Surface Normals
===============

Many mapping operations in ``weitsicht`` return surface normals as part of
:py:class:`~weitsicht.MappingResultSuccess` (field ``normals``).

Normals are returned as:

- a numpy array of shape ``(N, 3)``,
- **unit vectors** in the same CRS as ``coordinates``,
- ``nan`` for entries where ``mask`` is ``False``.

They are primarily used to refine perspective-image GSD estimation (see :doc:`../image/perspective_image`).

How normals are estimated (by mapper)
=====================================

The meaning of ``normals`` depends on the mapper backend:

.. list-table:: Normal estimation per mapper
   :header-rows: 1
   :widths: 28 72

   * - Mapper
     - Normal estimation
   * - ``MappingHorizontalPlane``
     - Constant upward normal ``(0, 0, 1)`` in mapper CRS (then transformed).
   * - ``MappingRaster`` (disk-backed)
     - Currently returns the local "up" direction of the mapper CRS (i.e. ``(0, 0, 1)`` transformed back to the source
       CRS). It does **not** encode DEM slope yet.
   * - ``MappingGeorefArray`` (in-memory)
     - Same as above: currently returns the local "up" direction of the mapper CRS.
   * - ``MappingTrimesh``
     - Uses the hit triangle's face normal (from ``trimesh``), oriented to face against the ray direction (towards the
       ray origin). If normals are unavailable/invalid, a fallback uses the **reverse ray direction** as an approximate
       normal.

CRS notes
=========

Normals are direction vectors and cannot be transformed by many CRS operations directly. When a CRS transform is
required, ``weitsicht`` approximates vector transformation by:

1. transforming a start point ``p`` and an end point ``p + v_unit * 10``, and
2. re-normalizing the transformed difference vector.

This works best in cartesian-like CRSs with linear units (meters), such as projected CRSs or ECEF.
For geographic CRSs (lon/lat degrees) the concept of "unit vectors" and angles becomes ambiguous; treat normals and
any derived quantities (like incidence-angle corrections) with care.

.. note::
   Raster-based mappers currently approximate normals as a purely vertical "up" vector. In many geographic 3D CRSs
   (lon/lat/height, e.g. WGS84 3D / ``EPSG:4979``), the height axis is a linear unit (typically meters), so transforming
   a purely vertical vector is often stable even though the horizontal axes are in degrees.

   This only addresses the *vector transform* itself; it does **not** make angle- or distance-based interpretations in a
   geographic CRS physically meaningful. For measurements (GSD, areas, incidence angles), prefer a projected metric CRS.
