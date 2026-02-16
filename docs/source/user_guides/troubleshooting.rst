===============
Troubleshooting
===============

Common issues
-------------

- **Image is not geo-referenced**: ``ImagePerspective`` requires ``camera``, ``position``, ``orientation``, and ``crs``.
  Missing geo-reference raises ``NotGeoreferencedError``.
- **Mapper missing**: mapping functions (``map_points``, ``map_footprint``, ``map_center_point``) require a mapper attached
  to the image or passed to the call. Missing mapper raises ``MapperMissingError``.
- **No intersections found**:
  - plane/raster mappers may return ``Issue.WRONG_DIRECTION`` when rays are parallel or point away,
  - raster/array mappers return ``Issue.OUTSIDE_RASTER`` when queries leave the raster extent,
  - raster mappers may return ``Issue.RASTER_NO_DATA`` when the ray touches no-data cells (holes) in the raster,
  - disk-backed raster ray intersection may return ``Issue.MAX_ITTERATION`` when the iterative solve does not converge,
  - mesh mapper can return ``Issue.NO_INTERSECTION`` when no triangles are hit.
- **Projection returns no valid pixels**: ``ImagePerspective.project`` can fail with
  ``Issue.INVALID_PROJECTIIONS`` when all projected points lie outside the distortion validity border.
- **CRS problems**: pass ``crs_s`` when projecting/mapping data in a different CRS; ensure your CRSs are truly 3D when
  heights matter; compound CRSs may require PROJ grids.

Debug helpers
-------------

- Log ``result.issues`` even when ``ok`` is ``True`` to spot partial validity.
- Enable logging (``logging.basicConfig(level=logging.DEBUG)``) to surface pyproj and mapper diagnostics.
- Inspect the distortion validity border via ``camera.distortion_border`` and test points with
  ``camera.undistorted_image_points_inside(...)``.
- For batch processing, use ``ImageBatch.project(..., only_valid=True)`` to find which images actually see target points.
