.. _image_ortho:

===========
Ortho Photo
===========

An orthophoto is a geometrically rectified image providing an orthographic view, where for each pixel the geolocation
is known. Orthophotos are typically produced from satellite imagery or aerial photography
(`Orthophoto <https://en.wikipedia.org/wiki/Orthophoto>`__).

In ``weitsicht`` the orthophoto implementation is :py:class:`weitsicht.ImageOrtho` and is based on
`rasterio <https://github.com/rasterio/rasterio>`__.

.. figure:: /_static/orthophoto.jpg
   :align: center
   :alt: orthophoto

   Orthophoto from Vienna showing parts of Schloss Schönbrunn.

Geo-reference
=============

The geo-reference can be provided either:

- as metadata in the file (e.g. GeoTIFF), or
- as sidecar files (e.g. ``.jgw`` for a JPG world file).

In GDAL terminology this is the `geotransform <https://gdal.org/en/stable/tutorials/geotransforms_tut.html>`__, an affine
transformation from pixel space (row/column) to a geo-referenced coordinate space.

See `World File <https://en.wikipedia.org/wiki/World_file>`__ for more background.

.. warning::
   Currently only the affine transformation (GDAL geotransform / world file) is implemented for ortho photos.
   RPCs and GCP-based geo-referencing are not supported.

Coordinate Reference System
============================

The CRS of an orthophoto is usually stored as metadata (GeoTIFF) and can sometimes also be supplied by a ``.prj`` sidecar
file containing WKT.

Projected and geographic CRSs are supported. For more information see
`WKT <https://en.wikipedia.org/wiki/Well-known_text_representation_of_coordinate_reference_systems>`__.

Project (3D → pixels)
=====================

To get the pixel location of a 2D/3D point, use ``ImageOrtho.project``.

1. Coordinates are transformed to the orthophoto CRS (if needed).
2. The affine transform is inverted to map ground coordinates into pixel space.

.. image:: /_static/orthop_project.svg
   :align: center
   :alt: orthophoto project

Mapping (pixels → 3D)
=====================

For orthophotos, mapping uses the orthophoto’s geotransform for XY and relies on the attached mapper (if any) to provide Z
via ``map_heights_from_coordinates``.

.. image:: /_static/orthop_z_mapper.svg
   :align: center
   :alt: orthophoto mapping

GSD and area (resolution-based)
===============================

Unlike perspective images, orthophotos are already rectified into an orthographic view. ``weitsicht`` therefore treats
their pixel size as constant and uses the dataset ``resolution`` (derived from the affine geo-transform):

- mapping results set ``gsd`` to ``ImageOrtho.resolution`` and fill ``gsd_per_point`` with that constant value,
- ``map_footprint`` computes ``area`` from the mapped footprint polygon in the orthophoto CRS.

.. warning::
   ``resolution``, ``gsd``, and the computed polygon ``area`` are expressed in the orthophoto CRS units. If the orthophoto
   CRS is geographic (lon/lat degrees) or otherwise non-metric, these values are degrees/pixel and degrees^2 (or other
   non-metric units) and are **not** meaningful physical measurements. Reproject the orthophoto to a projected metric CRS
   (meters) before using GSD/area for measurement workflows.
