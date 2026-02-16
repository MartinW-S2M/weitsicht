.. _horizontal_plane:

=======================
Horizontal Plane Mapper
=======================

.. hint::
   Even if in the mapper CRS all Z-coordinates are the same, transforming results back to the source CRS can change Z.
   Different ellipsoids and vertical shift grids (geoids) can apply. See :ref:`pyproj-hints`.

-------------------
CRS and projections
-------------------

This mapper can be used with most projected 3D CRSs (UTM, Lambert Conformal Conic, national grids, ...). Coordinate
transformations are handled via :doc:`../transformation`.

When calling mapping methods, pass either ``crs_s=...`` (source CRS) or a pre-built ``transformer=...``.

.. important::
   The ray/plane intersection is computed with Euclidean math and a fixed plane normal ``(0, 0, 1)``.
   For physically meaningful results, the ray input CRS must be a meter-based 3D CRS where Z represents height.

   Do not intersect rays directly in lon/lat degrees; transform to a projected CRS first and convert back if needed.


----------------------------
Get heights from coordinates
----------------------------
``MappingHorizontalPlane`` provides constant-height results via
:py:meth:`weitsicht.MappingHorizontalPlane.map_heights_from_coordinates`.
Conceptually, it transforms input XY into the mapper CRS, injects a constant Z, then transforms back.

Following steps are performed:

#. Input coordinates are transformed to the mapper CRS.
#. The plane altitude (constant height) is assigned to the Z-component in the mapper CRS.
#. These coordinates in the mapper CRS with the changed Z-component are transformed back to the source CRS.

As long as the coordinate transformation is working this will always return coordinates.

This process assumes that for both systems (source CRS and mapper CRS) the normal vector is vertical in both systems at all coordinates.

.. figure:: /_static/mapper_horizontal_get.svg
     :width: 400
     :align: center
     :alt: Heights from coordinates workflow

     Workflow to get heights for coordinates from the horizontal mapper


--------------------------------------------
Get intersection of ray and horizontal plane
--------------------------------------------
Ray intersection is implemented in
:py:meth:`weitsicht.MappingHorizontalPlane.map_coordinates_from_rays`.
The ray is defined by a start point and a direction vector, both given in the same source CRS.

Currently this is implemented as following:

#. For the ray start position (source CRS), a corresponding plane point is derived using
   :py:meth:`weitsicht.MappingHorizontalPlane.map_heights_from_coordinates`.
#. That plane point and a normal vector ``(0, 0, 1)`` are used to compute a line/plane intersection.
#. If a CRS transformation is involved, the plane point is refined iteratively at the current intersection location
   to reduce small vertical inconsistencies introduced by vertical transformations (see :ref:`trafo_vertical`).
#. Parallelity is checked.
#. The intersection direction is checked; intersections “behind” the ray direction are rejected.

There is always an intersection between a line and a plane (unless they are parallel).

.. figure:: /_static/mapper_horizontal.svg
     :width: 400
     :align: center
     :alt: Ray intersection on a horizontal plane

     Ray intersection on a horizontal plane (conceptual).

.. note::
   Without a CRS transformation, the intersection is a single ray–plane solve.

   If a CRS transformation is involved and contains vertical grids/shifts, then a constant-height plane in the mapper CRS
   can map to a slightly varying height in the source CRS depending on XY. ``MappingHorizontalPlane`` therefore refines
   the plane point iteratively until the intersection stabilizes (currently 1 mm tolerance or 20 iterations).

Related pages
=============

- :doc:`../transformation`
- :doc:`raster`
- :doc:`mesh`
