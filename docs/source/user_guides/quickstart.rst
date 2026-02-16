==========
Quickstart
==========

A short, step-by-step path: choose a mapper, build a camera, create a geo-referenced image, project 3D points, map pixels.

What the code does
------------------
- Enables pyproj network so compound CRSs (with geoid/vertical grids) can resolve required PROJ data.
- Sets a horizontal-plane mapper at altitude 250 m in UTM 33 with vertical reference (EGM2008) for mapper and image.
- Defines a calibrated OpenCV pinhole camera.
- Specifies image pose (position + orientation) and its CRS.
- Instantiates ``ImagePerspective`` combining camera, CRS, pose, and mapper.
- Calls ``project`` to project 3D coordinates into the image.
- Calls ``map_points`` to cast rays through pixels and intersect them with the mapper, returning 3D coords and GSD.

Hints
-----
- You can specify different CRSs for image and mapper. ``weitsicht`` will estimate transformations via pyproj.
- Add distortion parameters (k1-k4, p1, p2) to ``CameraOpenCVPerspective`` if known; otherwise they default to zero.
- For terrain-aware mapping, use a DEM/mesh mapper instead of a plane.
- Replace hardcoded pose with values from metadata (see :doc:`metadata`) for drone imagery workflows.

1) Imports
----------

.. literalinclude::  ../../../examples/00_user_guide_walkthroug.py
   :language: python
   :start-after: # Importing
   :end-before:  # CRS

2) Coordinate system
--------------------

.. literalinclude::  ../../../examples/00_user_guide_walkthroug.py
   :language: python
   :start-after: # CRS
   :end-before:  # MAPPER

3) Ground model (mapper)
------------------------
Use a horizontal plane at a constant altitude as the intersection surface.

.. literalinclude::  ../../../examples/00_user_guide_walkthroug.py
   :language: python
   :start-after: # MAPPER
   :end-before:  # CAMERA

4) Camera model
---------------
Provide intrinsics and distortion. Width/height belong to the image.

.. literalinclude::  ../../../examples/00_user_guide_walkthroug.py
   :language: python
   :start-after: # CAMERA
   :end-before:  # IMAGE

5) Geo-referenced image
-----------------------
Attach pose, CRS, camera, and mapper. Once set, ``image.is_geo_referenced`` is ``True``.

.. literalinclude::  ../../../examples/00_user_guide_walkthroug.py
   :language: python
   :start-after: # IMAGE
   :end-before:  # PROJECT

6) Project world points to pixels
---------------------------------
``project`` transforms coordinates to the image CRS, applies pose + intrinsics, and returns pixels + a validity mask.

.. literalinclude::  ../../../examples/00_user_guide_walkthroug.py
   :language: python
   :start-after: # PROJECT
   :end-before:  # MAPPING

7) Map pixels back to 3D
------------------------
``map_points`` casts rays through the pixels and intersects them with the mapper.

.. literalinclude::  ../../../examples/00_user_guide_walkthroug.py
   :language: python
   :start-after: # MAPPING
   :end-before:  # CHECK

8) Inspect issues
-----------------
Both ``proj`` and the mapping result carry ``issues`` and an ``ok`` flag—log them to catch partial failures.

.. literalinclude::  ../../../examples/00_user_guide_walkthroug.py
   :language: python
   :start-after: # CHECK

What to try next
----------------
- Swap the mapper for ``MappingRaster`` or ``MappingTrimesh`` for real terrain/mesh intersections.
- Batch process many images with ``ImageBatch`` and a shared mapper.
- See :doc:`results_prj_mapping` for return-object fields and issue codes.
