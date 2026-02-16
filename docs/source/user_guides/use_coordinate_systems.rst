Working with Coordinate Systems
===============================

Why this matters
----------------
Coordinate systems are inherently complex: projected grids (UTM, national grids), geocentric Cartesian (ECEF),
geodetic ellipsoidal (longitude/latitude/height), vertical datums, and time-dependent frames all have different units,
axis conventions, and distortion behavior.

In ``weitsicht`` this matters because:

- exterior orientation (camera pose) is expressed in an image CRS,
- your ground model (mapper) may live in a different CRS,
- your target coordinates may be in yet another CRS.

How ``weitsicht`` handles coordinates
-------------------------------------

``weitsicht`` uses :py:class:`weitsicht.transform.coordinates_transformer.CoordinateTransformer` (a thin wrapper around
pyproj) and follows a few rules:

- Minimal transformations: ``CoordinateTransformer.from_crs(crs_s, crs_t)`` returns ``None`` if both CRS are ``None`` or
  equal.
- Validation: supplying only one CRS raises ``ValueError``; both must be set or both ``None``.
- Axis order: transformers are created with ``always_xy=True`` (x/lon first).
- Best/ballpark settings:
  ``weitsicht.cfg._ballpark_transformation`` and ``weitsicht.cfg._only_best_transformation`` influence which pyproj
  pipelines are allowed.

Z axis / vertical reference
---------------------------

Most workflows require 3D CRS definitions with an explicit vertical reference. When a mapper CRS is set, mappers validate
that it has a Z axis; otherwise 3D transformations will not work and initialization can fail (e.g. with
``CRSnoZaxisError``).

Creating a transformer
----------------------

- CRS to CRS (typical): ``CoordinateTransformer.from_crs(crs_s, crs_t)``
- PROJ pipeline (advanced): ``CoordinateTransformer.from_pipeline("...proj pipeline string...")``

Transform direction is controlled via ``direction="forward"`` or ``direction="inverse"`` when calling
``transform(...)``.

Passing CRS vs passing a transformer
------------------------------------

Most high-level functions accept either:

- ``crs_s=...`` (source CRS), or
- ``transformer=...`` (a pre-built ``CoordinateTransformer``)

Do not pass both at the same time.

This is useful when you:

- project/map many point sets with the same CRS pair and want to reuse the transformer,
- want to enforce a specific PROJ pipeline for a regulated workflow.

PROJ grids and network access
-----------------------------

Some CRS transforms (especially involving vertical datums) require PROJ grid files. If those grids are missing, pyproj
may fail or fall back to less accurate “ballpark” transforms.

If you use compound CRSs and are okay with network downloads in your environment, enable:

.. code-block:: python

   import pyproj.network
   pyproj.network.set_network_enabled(True)

You can also tune the transformation policy:

.. code-block:: python

   import weitsicht
   weitsicht.allow_ballpark_transformations(True)
   weitsicht.allow_non_best_transformations(True)

Best-practice scenarios
-----------------------

- Same CRS everywhere: set CRSs to ``None`` where possible to skip transformations.
- Different projected CRSs but same geodetic frame: transforms are usually single-step and fast.
- Different frames / historic grids / vertical datums: expect grid requirements; validate with control points.
