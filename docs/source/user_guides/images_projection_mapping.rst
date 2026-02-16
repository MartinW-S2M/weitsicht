=======================
Projections and Mapping
=======================

``ImagePerspective`` and ``ImageOrtho`` wrap geo-referenced imagery and expose projection + mapping helpers.

Core concepts
-------------
- Projection = 3D world -> image pixels (``project``).
- Mapping = image pixels -> 3D coordinates (``map_points``, ``map_center_point``, ``map_footprint``).
- Monoplotting (see :doc:`/documentation/monoplotting`) = single-image mapping via ray/surface intersection (pixel -> ray -> ground model/mesh hit). In ``weitsicht`` this is exactly what the ``map_*`` methods do when a mapper (plane/raster/mesh/array) is provided.
- CRS matters: both directions transform inputs into the image CRS before optics/geo math.
- Mapper matters: mapping calls need a ground model (plane, raster, mesh, etc.).

Projection functions
--------------------
- ``ImagePerspective.project``: 3D coordinates -> pixels. Does CRS transform, applies pose, projects with camera model, re-applies distortion (only where inside distortion border). Needs ``camera``, ``position``, ``orientation``, ``crs`` set (``is_geo_referenced``).
- ``ImageOrtho.project``: 3D coordinates -> pixels via affine inverse transform; validity check is raster bounds.
- ``ImagePerspective.pixel_to_ray_vector``: pixels -> unit rays in camera CRS (optionally undistorted). Useful for custom ray tracing or mapper reuse.
- Validity: both return a mask marking in-frame pixels; outside-frame points stay in the array but are flagged.

Mapping functions
-----------------
- ``map_points`` (both classes): pixels -> 3D using the attached/provided mapper. Perspective computes GSD from range & focal length; ortho uses its ``resolution``.
- ``map_footprint`` (both): builds image footprint polygon in 3D; perspective can densify edges, ortho follows raster corners.
- ``map_center_point`` (both): maps principal point (perspective) or raster center (ortho) to 3D.
- Mapper requirements: a ``MappingBase`` instance must be present on the image or passed into the call. Typical choices: horizontal plane, raster DEM, mesh.
- CRS handling: pixels/rays are intersected in mapper CRS; inputs are transformed automatically via ``CoordinateTransformer``.

How projection works (perspective)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Transform coordinates from ``crs_s`` to image CRS.
2. Move into camera space with pose (orientation + position).
3. Project through camera intrinsics; undistort if requested.
4. Mark pixels outside distortion border as invalid.

How mapping works (pixels -> 3D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. (Perspective only) Pixels -> rays via ``pixel_to_ray_vector``; origins = image position.
2. (Ortho) Affine transform pixels to ground 2D, then mapper supplies heights if available.
3. Mapper intersects rays/points with its surface (plane/raster/mesh/array).
4. GSD derived from range (perspective) or stored resolution (ortho).

Example: project then map
-------------------------
.. code-block:: python

    import numpy as np
    from pyproj import CRS
    from weitsicht.image.perspective import ImagePerspective
    from weitsicht.mapping.horizontal_plane import MappingHorizontalPlane

    # assume camera is already built
    mapper = MappingHorizontalPlane(plane_altitude=5.0, crs=CRS.from_epsg(4979))
    img = ImagePerspective(width=4000, height=3000,
                           camera=camera,
                           crs=CRS("EPSG:31256+5778"),
                           position=np.array([0, 0, 120]),
                           orientation=Rotation.from_opk_degree(omega=0.0, phi=0.1, kappa=90.0),
                           mapper=mapper)

    pts_world = np.array([[5, 20, 0], [10, 25, 0]])
    proj = img.project(pts_world, crs_s=img.crs)
    pixels = proj.pixels[proj.mask]

    mp = img.map_points([[1000, 900], [1200, 1100]])
    coords_3d = mp.coordinates[mp.mask]
    gsd = mp.gsd

Tips
----
- ``project`` returns undistorted pixels when ``to_distorted`` is False; distortion is re-applied only where ``valid_mask`` is True.
- Use ``image_points_inside`` (perspective) or ``ImageOrtho.image_points_inside`` to pre-filter coordinates.
- For single points, both projection and mapping accept 1D arrays; they are internally promoted to Nx2 / Nx3.
- Keep mapper CRS aligned with image CRS to avoid extra transforms (or always pass ``crs_s``).
- For large point sets, batch your calls per image type to reuse transformers and mappers efficiently.
