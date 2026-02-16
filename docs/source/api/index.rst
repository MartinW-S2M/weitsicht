:html_theme.sidebar_secondary.remove: true

==================
API Reference
==================

.. currentmodule:: weitsicht

API Documentation

.. toctree::
   :maxdepth: 1
   :caption: General
   :hidden:
   :glob:

   alias
   return_types_info
   exceptions

.. toctree::
   :maxdepth: 1
   :caption: Transform
   :hidden:
   :glob:

   transform/*

.. toctree::
   :maxdepth: 1
   :caption: Camera Perspective
   :hidden:
   :glob:

   camera/*

.. toctree::
   :maxdepth: 1
   :caption: Image
   :hidden:
   :glob:

   image/*

.. toctree::
   :maxdepth: 1
   :caption: ImageBatch
   :hidden:
   :glob:

   ImageBatch

.. toctree::
   :maxdepth: 1
   :caption: Mapping
   :hidden:
   :glob:

   mapping/*

.. toctree::
   :maxdepth: 1
   :caption: Meta-Data
   :hidden:
   :glob:

   meta-data/*

General
===============
.. grid:: 2 2 3 3
    :gutter: 1

    .. grid-item-card:: Type Aliases
        :link: alias
        :link-type: doc

        Some type aliases for numpy arrays.

    .. grid-item-card:: Return Types
        :link: return_types_info
        :link-type: doc

        Return types for mapping and projections

    .. grid-item-card:: Exceptions
        :link: exceptions
        :link-type: doc

        weitsicht specific exceptions


Rotation & Coordinate Transformation
=====================================
.. grid:: 2 2 3 3
    :gutter: 1

    .. grid-item-card:: :py:class:`Rotation`
        :link: transform/Rotation
        :link-type: doc

        Create Rotation matrix to use for perspective images

    .. grid-item-card:: UTM converter
        :link: transform/utm_converter
        :link-type: doc

        Functions to derive utm coordinates with EGM2008 heights.

    .. grid-item-card:: :py:class:`CoordinateTransformer`
        :link: transform/coordinate_trafo
        :link-type: doc

        Helper class for coordinate transformation


Camera
===============
.. grid:: 2 2 3 3
    :gutter: 1

    .. grid-item-card:: :py:class:`CameraBasePerspective`
        :link: camera/CameraBasePerspective
        :link-type: doc

        Base Camera class

    .. grid-item-card:: :py:class:`CameraOpenCVPerspective`
        :link: camera/CameraOpenCVPerspective
        :link-type: doc

        Class using OpenCV definitions.


Images
===============
.. grid:: 2 2 3 3
    :gutter: 1

    .. grid-item-card:: :py:class:`ImageBase`
        :link: image/ImageBase
        :link-type: doc

        Base class for images

    .. grid-item-card:: :py:class:`ImagePerspective`
        :link: image/ImagePerspective
        :link-type: doc

        Perspective images (e.g. digital camera images)

    .. grid-item-card:: :py:class:`ImageOrtho`
        :link: image/ImageOrtho
        :link-type: doc

        Ortho photo mapping and projection


ImageBatch
===============
.. grid:: 2 2 3 3
    :gutter: 1

    .. grid-item-card:: :py:class:`ImageBatch`
        :link: ImageBatch
        :link-type: doc

        Container class for multiple images.

Mapper
===============
.. grid:: 2 2 3 3
    :gutter: 1

    .. grid-item-card:: :py:class:`MappingBase`
        :link: mapping/MappingBase
        :link-type: doc

        Base class for mapper

    .. grid-item-card:: :py:class:`MappingHorizontalPlane`
        :link: mapping/MappingHorizontalPlane
        :link-type: doc

        Minimal mapper


    .. grid-item-card:: :py:class:`MappingRaster`
        :link: mapping/MappingRaster
        :link-type: doc

        using raster data which can be loaded by ``rasterio``.


    .. grid-item-card:: :py:class:`MappingGeorefArray`
        :link: mapping/MappingGeorefArray
        :link-type: doc

        advanced raster mapper using bilinear intersection

    .. grid-item-card:: :py:class:`MappingTrimesh`
        :link: mapping/MappingTrimesh
        :link-type: doc

        using mesh data which can be loaded by ``trimesh``.


Meta-Data
===============
.. grid:: 2 2 3 3
    :gutter: 1

    .. grid-item-card:: Alternative Calibration Tags
        :link: meta-data/AlternativeCalibrationTags
        :link-type: doc

        :py:class:`AlternativeCalibrationTags` – vendor-specific calibration lookup.

    .. grid-item-card:: Camera Estimator
        :link: meta-data/camera_estimator_metadata
        :link-type: doc

        Functions to derive camera intrinsics from resolved metadata tags.

    .. grid-item-card:: Exterior Orientation
        :link: meta-data/eor_from_meta
        :link-type: doc

        :py:func:`eor_from_meta` – build EOR from metadata.

    .. grid-item-card:: Image From Meta
        :link: meta-data/image_from_meta
        :link-type: doc

        Build :py:class:`ImagePerspective` from tags (returns a result object).

    .. grid-item-card:: Metadata Results
        :link: meta-data/metadata_results
        :link-type: doc

        Return structures and issue codes for metadata extraction.

    .. grid-item-card:: PyExifTool Tags
        :link: meta-data/pyexiftool_tags
        :link-type: doc

        Resolver mapping Phil Harvey exiftool output to Meta tags.

    .. grid-item-card:: Tag Base Types
        :link: meta-data/tag_base
        :link-type: doc

        Core dataclasses and base parser interface.

    .. grid-item-card:: Camera Database
        :link: meta-data/camera_database
        :link-type: doc

        Sensor size lookup and helper utilities.
