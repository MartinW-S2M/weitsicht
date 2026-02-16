==============
Error Handling
==============

``weitsicht`` uses a mix of:

- **Exceptions** for invalid inputs and missing prerequisites
- **Result objects** (``ProjectionResult`` / ``MappingResult``) for runtime outcomes like "no intersection"

This guide explains which is which, and how to handle them in user code.


Two kinds of failures
=====================

1) Exceptions (you need to fix inputs / state)
----------------------------------------------

Exceptions are used when the function cannot even *attempt* the operation, when inputs are inconsistent, or when a
core dependency (like coordinate transformation) fails:

- ``NotGeoreferencedError``: required pose/camera/georeference is missing
- ``MapperMissingError``: mapping is requested but no mapper is attached/passed
- ``CRSInputError``: inconsistent CRS arguments (e.g. providing both ``crs_s`` and ``transformer``)
- ``CoordinateTransformationError``: a coordinate transformation could not be established or applied

``CRSInputError`` is also a ``ValueError`` (multiple inheritance), so ``except ValueError`` will catch it as well.


2) ResultFailure (the operation ran but produced no/partial output)
-------------------------------------------------------------------

Most projection/mapping operations return a result object:

- projection: ``ProjectionResult`` = ``ProjectionResultSuccess`` | ``ResultFailure``
- mapping: ``MappingResult`` = ``MappingResultSuccess`` | ``ResultFailure``

Typical "runtime outcome" failures (no exception) include:

- rays point in the wrong direction
- intersections fall outside a raster/array extent
- no intersection exists for a ray

In these cases you get ``ResultFailure(ok=False, error=..., issues=...)`` or a success result with a partial ``mask``.

See :doc:`results_prj_mapping` for the result fields and mask semantics.


Exception hierarchy (current)
=============================

All package-specific exceptions inherit from ``WeitsichtError``.

- ``WeitsichtError``

  - ``CRSInputError`` (also ``ValueError``)

    - ``CRSnoZaxisError``

  - ``CoordinateTransformationError``
  - ``NotGeoreferencedError``

  - ``MappingError``

    - ``MapperMissingError``
    - ``MappingBackendError``

Notes:

- ``CoordinateTransformationError`` is intentionally *not* wrapped into ``MappingError`` so you can catch it explicitly
  across the whole package.
- ``MappingBackendError`` is raised by image ``map_*`` methods when the mapper backend throws an unexpected, non-weitsicht
  exception. The original exception is available via ``err.__cause__``.


Recommended try/except patterns
===============================

Catch only what you can handle (mapping)
----------------------------------------

.. code-block:: python

   from weitsicht import (
       CoordinateTransformationError,
       CRSInputError,
       MapperMissingError,
       MappingBackendError,
       MappingError,
       NotGeoreferencedError,
   )

   try:
       mp = image.map_points([[200, 300], [50, 50]])
   except CRSInputError:
       raise
   except CoordinateTransformationError:
       raise
   except (NotGeoreferencedError, MapperMissingError):
       raise
   except MappingBackendError as err:
       # unexpected backend exception from the mapper class; the root cause is in err.__cause__
       raise

   if not mp.ok:
       print("Mapping failed:", mp.error, "Issues:", mp.issues)
   else:
       coords_valid = mp.coordinates[mp.mask]


Catch only what you can handle (projection)
-------------------------------------------

.. code-block:: python

   from weitsicht import CoordinateTransformationError, CRSInputError, NotGeoreferencedError

   try:
       proj = image.project(points_3d, crs_s=points_crs)
   except CRSInputError:
       # invalid CRS / transformer usage (e.g. crs_s and transformer both set)
       raise
   except CoordinateTransformationError:
       # transformation could not be established or applied
       raise
   except NotGeoreferencedError:
       raise

   if not proj.ok:
       print("Projection failed:", proj.error, "Issues:", proj.issues)
   else:
       pixels_valid = proj.pixels[proj.mask]


Working with ImageBatch
=======================

``ImageBatch`` methods return per-image results (dicts of ``MappingResult`` / ``ProjectionResult``).
They catch common per-image errors (for example ``NotGeoreferencedError``, ``MapperMissingError``, ``CRSInputError``,
``MappingError``) and convert them into ``ResultFailure`` with ``Issue.IMAGE_BATCH_ERROR`` so other images can still be
processed.

Some exceptions (notably ``CoordinateTransformationError``) may still propagate to the caller so you can handle
transformation problems explicitly.


Related guides
==============

- :doc:`results_prj_mapping`
- :doc:`use_coordinate_systems`
- :doc:`troubleshooting`
