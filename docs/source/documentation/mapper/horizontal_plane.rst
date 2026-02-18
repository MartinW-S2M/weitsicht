.. _horizontal_plane:

=======================
Horizontal Plane Mapper
=======================

.. note:: In the near future there will be a more generall class for planes where plane point and normal vector are used for initialization.
   ``MappingHorizontalPlane`` will still be possible to be used and is keept for legacy reasons.

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
   For physically meaningful results, the ray input CRS must be a meter-based 3D CRS.
   Do not form rays directly in lon/lat degrees; transform to a projected CRS first and convert back if needed.


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
   :py:meth:`weitsicht.MappingHorizontalPlane.map_heights_from_coordinates` in the mapper crs.
#. That plane point and a normal vector ``(0, 0, 1)`` are transformed back to the source crs-
#. Line/plane intersection is computed.
#. The intersection direction is checked; Parallel rays or intersections “behind” the ray direction are rejected.
#. Iterate over steps but use the now intersection point instead of start position for the plane point and normals

There is always an intersection between a line and a plane (unless they are parallel).

.. figure:: /_static/mapper_horizontal.svg
     :width: 400
     :align: center
     :alt: Ray intersection on a horizontal plane

     Ray intersection with a plane (conceptual).

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
