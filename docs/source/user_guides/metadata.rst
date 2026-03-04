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

.. important::
   The estimated image pose (EOR: position + orientation) is returned in **WGS84 geocentric / ECEF**
   (``EPSG:4978``) by default (i.e. unless you set ``to_utm=True``). You can always inspect the CRS from the result
   (``eor_res.crs``) or, when building a full image, via ``image.crs``.

CRS override (optional)
^^^^^^^^^^^^^^^^^^^^^^^^

By default, ``eor_from_meta``, ``image_from_meta`` and ``ImageBuilder.eor()`` tries to interpret the position tags using CRS information found in metadata (custom
XMP tags like ``HorizCS``/``VertCS``). If those tags are missing, it falls back to WGS84 (typically ``EPSG:4979`` for
ellipsoidal heights; ``EPSG:4326+3855`` for orthometric/relative modes).

You can override the CRS detection by passing ``crs=...``. This is useful when:

- the CRS tags are missing or wrong,
- you want to enforce a specific vertical datum (use a 3D CRS or a compound CRS),
- you inject positions from RTK/DGPS processing that are referenced to a known CRS from your base/reference setup.

.. code-block:: python

   from pyproj import CRS
   from weitsicht import eor_from_meta

   # Example: ETRS89 ECEF coordinates (adapt EPSG to your reference frame if needed)
   crs_rtk = CRS.from_epsg(4936)

   eor_res = eor_from_meta(tags.get_all(), crs=crs_rtk)

.. important::
   Even if you pass a ``crs`` to the metadata helpers, the pose is still converted to **WGS84 ECEF**
   (``EPSG:4978``). The reason is that we assume the orientation angles describe a **local tangent plane** at the image
   position. Converting the position to ECEF first and then deriving the local tangent frame is robust and keeps the
   implementation simple. Building the local tangent plane directly in an arbitrary CRS would require deeper analysis
   of the CRS datum/ellipsoid and axis conventions.

.. important::
   If you work with a specific CRS **realization** (e.g. a particular ETRS89/ETRF frame or a national realization),
   you must use the correct EPSG code for that system/realization. To avoid subtle offsets, make sure that for your mapper
   (rasters, meshes, reference layers) you also use then the correct **realization/datum** that mapper data was derived from.

   This is a broad topic and we will not further at the moment cover that more specific.

   Time-dependent transformations (e.g. plate motion) are currently not considered because the CoordinateTransformer do not
   accept a coordinate's epoch/time. If you need epoch-aware transformations, use ``pyproj`` directly.

Vertical reference (Z) options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 ``eor_from_meta``, ``image_from_meta`` and ``ImageBuilder.eor()`` support different ways to interpret the height component:

- ``vertical_ref="ellipsoidal"`` (default): interpret metadata altitude as **ellipsoidal height** (WGS84).
- ``vertical_ref="orthometric"``: interpret metadata altitude as **orthometric/geoid height** (EGM2008).
- ``vertical_ref="relative"``: use a vendor ``RelativeAltitude`` tag (if present) and compute
  ``z = height_rel + RelativeAltitude``.

``height_rel`` is only used for ``vertical_ref="relative"`` and should be set to the reference height (meters) that the
relative altitude is measured from (often the take-off height, if known).

UTM output (optional)
^^^^^^^^^^^^^^^^^^^^^^

By default, ``eor_from_meta`` returns the pose in WGS84 ECEF (``EPSG:4978``). If you prefer a local projected output,
set ``to_utm=True``. The UTM zone is chosen automatically from the WGS84 lon/lat:

.. code-block:: python

   from weitsicht import eor_from_meta

   eor_res = eor_from_meta(tags.get_all(), to_utm=True)
   if eor_res.ok:
       print(eor_res.crs)  # e.g. EPSG:32633+3855 (zone depends on lon/lat)
       utm_position = eor_res.position
       utm_orientation = eor_res.orientation  # aligned to the UTM grid (meridian convergence applied)
   else:
       print(eor_res.error, eor_res.issues)

.. note::
   ``to_utm=True`` outputs a compound CRS (UTM + EGM2008 height, ``+3855``). This can require PROJ grid data.
   If you see missing-grid errors, enable network grids via ``pyproj.network.set_network_enabled(True)`` (see :doc:`top_tips`).


Complete image from metadata
----------------------------

.. code-block:: python

   from weitsicht import image_from_meta

   img_res = image_from_meta(tags)  # ECEF output (default)
   # img_res = image_from_meta(tags, to_utm=True)  # projected UTM output
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
   # eor_res = builder.eor(to_utm=True)

   # Optionally build an image only if both are available
   img_res = builder.image()

Notes & tips
------------

- **Sensor database fallback** (``src/weitsicht/metadata/camera_database.py``) supplies sensor sizes for common models
  when EXIF lacks focal-plane resolutions.
- **CRS is separate**: metadata rarely contains a complete CRS definition; pass a ``pyproj.CRS`` explicitly if needed.
- **Validation**: after building an image, map a footprint and compare it to known GIS layers to detect convention issues.
