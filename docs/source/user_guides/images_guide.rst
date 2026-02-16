============
Images Guide
============

This guide explains how perspective images are handled in ``weitsicht`` and how that practical workflow
maps to the theory in :doc:`/documentation/image/perspective_image`.

Scope
-----
A perspective image in ``weitsicht`` is used as a geometric object that combines:

- image pixel coordinates,
- a camera model (intrinsics + distortion),
- exterior orientation (camera pose),
- optional mapper context for ray intersection.

If one of these parts is missing, some methods still run, but geo-accurate mapping results are limited.

The pixel → 3D intersection step for a *single* image is often referred to as **Monoplotting** (see :doc:`/documentation/monoplotting`).

Theory to Practice Mapping
--------------------------
The theory page defines:

- projection model (3D -> pixel),
- inverse projection (pixel -> ray),
- need for a surface constraint to get a unique 3D point.

In ``weitsicht``, this maps to practical operations:

1. ``ImagePerspective.project(...)``:
   projects 3D points to image pixels (with optional ``to_distorted`` behavior).
2. ``ImagePerspective.pixel_to_ray_vector(...)``:
   converts image pixels to ray directions.
3. ``ImagePerspective.map_points(...)``:
   maps image pixels to 3D by ray intersection with a mapper.
4. ``ImagePerspective.map_center_point(...)``:
   maps the principal point to 3D.
5. ``ImagePerspective.map_footprint(...)``:
   maps corner/edge pixels to a 3D footprint.

Related camera-model conversion methods:

- ``camera.distorted_to_undistorted(...)``
- ``camera.undistorted_to_distorted(...)``
- ``camera.pts_camara_crs_to_image_pixel(...)``
- ``camera.pixel_image_to_camera_crs(...)``

See:

- :doc:`/documentation/image/perspective_image`
- :doc:`/documentation/image/camera`
- :doc:`/documentation/camera_crs`
- :doc:`/documentation/pixel_crs`
- :doc:`mappers`

Required Inputs
---------------
For reliable georeferenced image usage, provide:

- a valid camera model (currently OpenCV pinhole),
- valid image size matching camera calibration assumptions,
- valid EOR (position + orientation),
- clear CRS relation between EOR and target coordinates.

How Camera, Geo-reference, and Mapper Are Assigned
--------------------------------------------------
The perspective image object is ``ImagePerspective``.

Constructor-based assignment:

.. code-block:: python

    from pyproj import CRS
    from weitsicht.image.perspective import ImagePerspective
    from weitsicht.transform.rotation import Rotation

    image = ImagePerspective(
        width=6000,
        height=4000,
        camera=camera_obj,                       # CameraBasePerspective implementation
        crs=CRS("EPSG:31256+5778"),                # perspective image CRS
        position=[x, y, z],                      # EOR position in that CRS
        orientation=Rotation(rotation_matrix=R), # EOR orientation
        mapper=mapper_obj,                       # optional, can also be passed per call
    )

Dictionary-based assignment:

.. code-block:: python

    image = ImagePerspective.from_dict(
        {
            "width": 6000,
            "height": 4000,
            "position": [x, y, z],
            "orientation_matrix": R.tolist(),
            "camera": camera_obj.param_dict,
            "crs": CRS("EPSG:31256+5778").to_wkt(),
        },
        mapper=mapper_obj,
    )

Geo-reference validity in code:

- ``image.is_geo_referenced`` is ``True`` when position, orientation, and camera are set.
- Mapping methods also need a mapper (assigned on the image or passed to the call).

Practical Workflow
------------------
Typical usage order:

1. Load camera model parameters.
2. Build/load perspective image object with camera + EOR.
3. Validate coordinate conventions (camera CRS and pixel CRS).
4. Project a few known points and inspect residuals.
5. Convert image pixels to rays when needed.
6. Intersect rays with a mapper for 3D coordinates.
7. Use returned masks/issues to filter invalid outputs.

Method Summary (Actual API)
---------------------------
- ``project(coordinates, crs_s=None, transformer=None, to_distorted=True)``:
  3D -> pixel projection.
- ``pixel_to_ray_vector(pixel_pos, is_undistorted=False)``:
  image pixel -> ray vector.
- ``map_points(points_image, mapper=None, transformer=None, is_undistorted=False)``:
  pixel -> 3D coordinates through mapper intersection.
- ``map_center_point(mapper=None, transformer=None)``:
  principal point -> 3D.
- ``map_footprint(points_per_edge=0, mapper=None, transformer=None)``:
  image footprint -> 3D polygon points.
- ``image_points_inside(point_image_coordinates)``:
  checks if undistorted image points are inside the valid distortion border.

Distortion and Valid Pixels
---------------------------
If distortion is enabled, not every undistorted pixel is valid.
``weitsicht`` uses a precomputed distortion border in the camera model to mask invalid undistorted pixels.

For details, see :doc:`cameras_distortion`.

Common Pitfalls
---------------
- Using EOR from another software without converting convention.
- Mixing image size and calibration size without updating intrinsics.
- Ignoring validity masks from projection/mapping results.
- Assuming one CRS for all components when 3D point CRS and perspective image CRS differ.

Related User Guides
-------------------
- :doc:`rotation`
- :doc:`images_projection_mapping`
- :doc:`results_prj_mapping`
- :doc:`metadata`
- :doc:`cameras_distortion`
- :doc:`use_coordinate_systems`
