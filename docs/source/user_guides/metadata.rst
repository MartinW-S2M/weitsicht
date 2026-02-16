========================================
Metadata for Camera and Pose Extraction
========================================

``weitsicht`` turns EXIF/XMP tags into usable camera models and image parameters. The package is prepared to support
multiple metadata wrappers by defining a small tag interface (:class:`~weitsicht.metadata.tag_systems.tag_base.MetaTagsBase`).
Currently, ``weitsicht`` ships with :class:`~weitsicht.metadata.tag_systems.pyexiftool_tags.PyExifToolTags`, which works
with metadata dictionaries produced by `Phil Harvey's exiftool` (via CLI JSON output or Python wrappers).

Use the helpers as building blocks:

- retrieve IOR (intrinsics) for a camera model,
- extract EOR (position/orientation + CRS),
- build a ready-to-use ``ImagePerspective`` from metadata.

.. note::
   There is no universal metadata standard: vendors mix EXIF, XMP, and MakerNotes inconsistently. Standard GPS tags are
   often present but are insufficient without orientation tags. Always sanity-check by mapping an image footprint and
   comparing it with a known ortho or satellite layer before running large batches.

Minimal example
---------------

Extract tags (e.g. with **Phil Harvey's exiftool**) and wrap the resulting dictionary with ``PyExifToolTags``.

.. code-block:: python

   from pathlib import Path
   import json

   # Option A: Python helper (needs exiftool on PATH)
   from exiftool import ExifToolHelper
   with ExifToolHelper() as et:
       meta_list = et.get_metadata([Path("DJI_0001.JPG")])

   # Option B: CLI + JSON
   # import subprocess
   # raw = subprocess.check_output(["exiftool", "-json", "DJI_0001.JPG"])
   # meta_list = json.loads(raw)

   from weitsicht import PyExifToolTags
   tags = PyExifToolTags(meta_list[0])

Parse data (PyExifToolTags)
---------------------------

``PyExifToolTags`` provides convenient accessors that return typed dataclasses. Other metadata sources can be supported
by implementing :class:`~weitsicht.metadata.tag_systems.tag_base.MetaTagsBase` and returning the same tag dataclasses.

.. code-block:: python

   # image dimensions (width, height)
   tags.image_shape()

   # group all resolved tags into one object
   tags_all = tags.get_all()

   # camera/intrinsics related tags
   tags.get_ior_base()       # -> MetaTagIORBase
   tags.get_ior_extended()   # -> MetaTagIORExtended

   # GNSS and orientation tags
   tags.get_standard_gps()         # -> MetaTagGPS
   tags.get_orientation_values()   # -> MetaTagOrientation

   # optional vendor alternatives
   tags.get_z_alternatives()  # -> MetaTagZalternatives (e.g. XMP:RelativeAltitude)
   tags.get_crs()             # -> MetaTagCRS (custom HorizCS/VertCS tags)

Camera IOR options
------------------

Option 1 (baseline): estimate minimal intrinsics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Estimate camera intrinsics from standard tags: ``estimate_camera(tags.get_ior_base())`` returns
``width, height, focal_px, cx, cy``.

.. code-block:: python

   from weitsicht import CameraOpenCVPerspective, estimate_camera

   width, height, focal_px, cx, cy = estimate_camera(tags.get_ior_base())
   camera = CameraOpenCVPerspective(
       width=width,
       height=height,
       fx=focal_px,
       fy=focal_px,
       cx=cx,
       cy=cy,
   )

Option 2: use ``ior_from_meta`` (preferred)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ior_from_meta`` wraps estimation and also checks for extended/vendor calibration tags when present.

.. code-block:: python

   from weitsicht import ior_from_meta

   ior_res = ior_from_meta(
       tags_ior=tags.get_ior_base(),
       tags_ior_extended=tags.get_ior_extended(),
   )
   if ior_res.ok:
       camera = ior_res.camera
       width = ior_res.width
       height = ior_res.height
   else:
       # ResultFailure[MetadataIssue]
       print(ior_res.error, ior_res.issues)

Exterior orientation (EOR)
--------------------------

.. code-block:: python

   from weitsicht import eor_from_meta

   eor_res = eor_from_meta(tags.get_all())
   if eor_res.ok:
       position = eor_res.position
       orientation = eor_res.orientation
       crs = eor_res.crs
   else:
       print(eor_res.error, eor_res.issues)

Complete image from metadata
----------------------------

.. code-block:: python

   from weitsicht import image_from_meta

   img_res = image_from_meta(tags)  # accepts MetaTagsBase or MetaTagAll
   if img_res.ok:
       image = img_res.image
       # Assign a mapper later, e.g. image.mapper = MappingHorizontalPlane(...)
   else:
       print(img_res.error, img_res.issues)

Return structures
-----------------

The metadata helpers return small result objects:

- Success results are dataclasses with ``ok=True`` (e.g. ``IORFromMetaResultSuccess``).
- Failures are :class:`~weitsicht.utils.ResultFailure` with ``ok=False`` and the fields:
  ``error`` (human-readable message) and ``issues`` (a set of :class:`~weitsicht.metadata.metadata_results.MetadataIssue`).

The concrete unions are:

- ``IORFromMetaResult = IORFromMetaResultSuccess | ResultFailure[MetadataIssue]``
- ``EORFromMetaResult = EORFromMetaResultSuccess | ResultFailure[MetadataIssue]``
- ``ImageFromMetaResult = ImageFromMetaResultSuccess | ResultFailure[MetadataIssue]``

Keeping IOR and EOR decoupled
-----------------------------

If you want to handle IOR and EOR separately (e.g. use a default IOR but read EOR from IMU), use the builder:

.. code-block:: python

   from weitsicht.metadata.image_from_meta import ImageFromMetaBuilder

   builder = ImageFromMetaBuilder(tags)
   ior_res = builder.ior()
   eor_res = builder.eor()

   # Optionally build an image only if both are available
   img_res = builder.image()

Notes & tips
------------

- **Sensor database fallback** (``src/weitsicht/metadata/camera_database.py``) supplies sensor sizes for common models
  when EXIF lacks focal-plane resolutions.
- **CRS is separate**: metadata rarely contains a complete CRS definition; pass a ``pyproj.CRS`` explicitly if needed.
- **Validation**: after building an image, map a footprint and compare it to known GIS layers to detect convention issues.
