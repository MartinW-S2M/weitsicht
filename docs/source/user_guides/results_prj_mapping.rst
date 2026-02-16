==========================
Results, Masks, and Errors
==========================

For metadata functions (``ior_from_meta``, ``eor_from_meta``, ``image_from_meta``), the result structures and issue
codes are documented in :doc:`metadata`.

``weitsicht`` uses two result types for the main operations:

- **Projection** (3D → pixels): returns a ``ProjectionResult``
- **Mapping** (pixels → 3D): returns a ``MappingResult``

Both are union types:

- ``ProjectionResult`` = ``ProjectionResultSuccess`` | ``ResultFailure``
- ``MappingResult`` = ``MappingResultSuccess`` | ``ResultFailure``

Required inputs vs. runtime failures
------------------------------------

If required inputs are missing, functions raise exceptions (for example):

- ``NotGeoreferencedError``: a perspective image is missing camera/pose/CRS, or an ortho image lacks geotransform.
- ``MapperMissingError``: mapping is requested but no mapper is attached/provided.
- ``CRSInputError``: CRS information is required but missing/inconsistent.
- ``CoordinateTransformationError``: a CRS transformation fails (for example due to missing grids or invalid CRS).
- ``ValueError``: input arrays have an unexpected shape/dimension (for example not ``(N, 2)`` for pixels or not
  ``(N, 3)`` for coordinates).

If inputs are valid but the operation cannot produce valid output, functions return ``ResultFailure`` (or a success
result with a partial mask).

Success types (what they contain)
---------------------------------

``ProjectionResultSuccess``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``ok``: ``True``
- ``pixels``: ``np.ndarray`` of shape (N, 2)
- ``mask``: ``np.ndarray[bool]`` of length N
- ``issues``: ``set[Issue]`` (warnings; may be non-empty even on success)

``MappingResultSuccess``
^^^^^^^^^^^^^^^^^^^^^^^^

- ``ok``: ``True``
- ``coordinates``: ``np.ndarray`` of shape (N, 3)
- ``mask``: ``np.ndarray[bool]`` of length N
- ``crs``: output CRS (may be ``None``)
- optional metadata: ``gsd``, ``gsd_per_point``, ``area``
- ``issues``: ``set[Issue]`` (warnings)

Failure type
------------

``ResultFailure`` contains:

- ``ok``: ``False``
- ``error``: a string explanation
- ``issues``: ``set[Issue]``

Mask semantics
--------------

The mask always aligns with the input order:

- projection mask: pixels that are valid (inside image / distortion validity border),
- mapping mask: rays/points that produced a valid 3D intersection.

Many functions return a failure if **no** entries are valid, to make the problem explicit.

ImageBatch behavior
-------------------

Batch functions return per-image results:

- mapping: ``dict[str, MappingResult]``
- projection: ``dict[str, ProjectionResult]`` (or ``None`` when ``only_valid=True`` filters everything out)

Batch methods catch common per-image exceptions and return ``ResultFailure`` for that image with
``Issue.IMAGE_BATCH_ERROR`` set, so you can still process other images in the batch.

Issue catalogue (current)
-------------------------

``Issue`` values are defined in ``weitsicht.utils.Issue``:

- ``WRONG_DIRECTION``: a ray likely points away from the surface (or is incompatible with the mapper geometry).
- ``OUTSIDE_RASTER``: coordinates/rays fall outside raster/array extent.
- ``RASTER_NO_DATA``: mapping touched no-data raster cells (e.g. holes in the DEM), so no valid height/intersection exists.
- ``MAX_ITTERATION``: raster ray intersection reached the maximum iteration count (non-converging iterative solve).
- ``NO_INTERSECTION``: geometry intersection failed entirely (e.g. mesh hit test).
- ``INVALID_PROJECTIIONS``: projection produced no valid pixels inside the distortion validity border.
- ``IMAGE_BATCH_ERROR``: a single image in an ImageBatch call raised an error that was converted into a failure result.
- ``UNKNOWN``: fallback for unexpected conditions.

Handling results
----------------

.. code-block:: python

   mp = img.map_points([[200, 300], [50, 50]])
   if mp.ok:
       coords = mp.coordinates[mp.mask]
   else:
       print("Failed:", mp.error, "Issues:", mp.issues)

   proj = img.project(points_3d, crs_s=img.crs)
   if proj.ok:
       valid_px = proj.pixels[proj.mask]
       if proj.issues:
           print("Issues:", proj.issues)
   else:
       print("Failed:", proj.error, "Issues:", proj.issues)

Related guides
--------------

- :doc:`use_coordinate_systems`
- :doc:`mappers`
