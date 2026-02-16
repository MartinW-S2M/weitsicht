.. _transformation:

==========================
Coordinate Transformations
==========================

``weitsicht`` relies on coordinate reference systems (CRS) to combine geo-referenced imagery, navigation data,
and ground models (mappers). When CRSs differ, coordinates must be transformed consistently in **both**
mapping directions:

- **Projection**: 3D coordinates in a source CRS → image CRS → pixels.
- **Mapping**: pixels → rays in image CRS → mapper CRS → 3D intersection coordinates (often transformed back).

In the codebase, CRS handling is implemented via :py:class:`weitsicht.transform.coordinates_transformer.CoordinateTransformer`.

CoordinateTransformer (how it behaves)
======================================

``CoordinateTransformer.from_crs(crs_s, crs_t)`` returns:

- ``None`` if both CRSs are ``None`` (no CRS semantics) **or** if both CRSs are equal.
- a transformer chain otherwise.

This is intentional: when no transformation is required, skipping it avoids numeric noise and avoids depending on
external grid availability.

When a transform is required, ``from_crs`` may build either:

- a single CRS→CRS step (same geodetic CRS), or
- a **three-step** chain via geodetic CRSs (different geodetic CRS):

  1. source CRS → source geodetic CRS (3D)
  2. source geodetic CRS → target geodetic CRS (3D)
  3. target geodetic CRS → target CRS

This makes transformations more deterministic across different CRS families, at the cost of a few extra steps.

Passing transformations into weitsicht functions
=================================================

Most projection/mapping functions accept either:

- ``crs_s=...`` (source CRS), or
- ``transformer=...`` (a pre-built :py:class:`~weitsicht.transform.coordinates_transformer.CoordinateTransformer`)

Do not pass both at the same time.

This is useful when you:

- project many point sets into the same image and want to reuse the same transformer,
- already have a validated PROJ pipeline (``CoordinateTransformer.from_pipeline(...)``),
- want to force a specific transformation path for project requirements.

.. _pyproj-hints:

pyproj / PROJ grid hints
========================

Some transformations (especially those involving **vertical datums** or certain national grids) require PROJ grid files.
If those grids are missing, transformations may fail or fall back to less accurate “ballpark” pipelines.

Common tips:

- If your workflow uses compound CRSs (horizontal + vertical), enable network grid downloads in PROJ:

  .. code-block:: python

     import pyproj.network
     pyproj.network.set_network_enabled(True)

- Control whether approximate grids are allowed via:

  .. code-block:: python

     import weitsicht

     weitsicht.allow_ballpark_transformations(True)
     weitsicht.allow_non_best_transformations(True)

If your environment must run offline, pre-install the required PROJ data and disable network access.

.. _trafo_vertical:

Vertical transformations and “horizontal planes”
================================================

Several mappers (especially :py:class:`weitsicht.MappingHorizontalPlane`) conceptually operate on a plane of constant
height. When CRSs include different vertical references (ellipsoidal vs orthometric heights, different geoids, etc.),
“constant height” can become non-trivial across CRSs:

- assigning a constant ``z`` in the mapper CRS and transforming back to the source CRS can change ``z`` slightly,
  depending on the vertical pipeline,
- even when the surface is “flat” in one CRS, it may not be flat after a CRS change that applies vertical shifts.

If you see unexpected height differences:

- verify your CRS definitions are truly 3D and include the intended vertical datum,
- confirm grid availability (see :ref:`pyproj-hints`),
- prefer keeping image CRS and mapper CRS aligned when possible to reduce transform complexity.
