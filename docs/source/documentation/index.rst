:html_theme.sidebar_secondary.remove: true

.. _documentation:

==============
Documentation
==============
This section provides the conceptual and mathematical background for ``weitsicht``: coordinate systems, camera models,
image types, transformations, and mapper (ground model) assumptions.

.. toctree::
   :maxdepth: 1
   :caption: Basic Information
   :hidden:
   :glob:

   geo-reference_01
   monoplotting


.. toctree::
   :maxdepth: 1
   :caption: Definitions
   :hidden:
   :glob:

   pixel_crs
   camera_crs
   transformation
   pose_orientation



.. toctree::
   :maxdepth: 1
   :caption: Perspective Image
   :hidden:
   :glob:

   image/perspective_image
   image/camera


.. toctree::
   :maxdepth: 1
   :caption: Ortho Photo
   :hidden:
   :glob:

   ortho_image

.. toctree::
   :maxdepth: 1
   :caption: Mapper
   :hidden:
   :glob:

   mapper/*


Basics
-------------

.. grid:: 2 2 3 3
    :gutter: 1

    .. grid-item-card:: What is Geo-Referencing
        :link: geo-reference_01
        :link-type: doc

        Basics about geo-referenced images.

    .. grid-item-card:: Monoplotting
        :link: monoplotting
        :link-type: doc

        Single-image pixel → 3D mapping by ray–surface intersection.


Definitions
------------
.. important:: This section is critical on how input data need to be defined.

.. grid:: 2 2 3 3
    :gutter: 1

    .. grid-item-card:: Pixel CRS
        :link: pixel_crs
        :link-type: doc

        XY definition for pixels

    .. grid-item-card:: Camera CRS
        :link: camera_crs
        :link-type: doc

        Camera CRS definition used

    .. grid-item-card:: Pose / Rotation Angles
        :link: pose_orientation
        :link-type: doc

        Rotation matrix basics and common OPK / AZK (APK) notations.


Perspective Image
-----------------

.. grid:: 2 2 3 3
    :gutter: 1

    .. grid-item-card:: Basics
        :link: image/perspective_image
        :link-type: doc

        Basics on perspective images

    .. grid-item-card:: Camera Models
        :link: image/camera
        :link-type: doc

        Pinhole Model and Distortion


Ortho Imagery
-------------

.. grid:: 2 2 3 3
    :gutter: 1

    .. grid-item-card:: Ortho photos
        :link: ortho_image
        :link-type: doc

        Rectified image in orthographic view.

Mapper
-------------
| The main purpose of the mapper classes is to get the intersection of a ray.
| As well it can also be used to get the height of a coordinate.

.. grid:: 2 2 3 3
    :gutter: 1

    .. grid-item-card:: Horizontal plane
        :link: mapper/horizontal_plane
        :link-type: doc

        Constant-height plane (fast, flat approximation).

    .. grid-item-card:: Raster (DEM/DSM)
        :link: mapper/raster
        :link-type: doc

        Raster-based ground model (rasterio), terrain-aware intersections.

    .. grid-item-card:: GeorefArray (in-memory)
        :link: mapper/georef_array
        :link-type: doc

        In-memory raster window backend with bilinear patch intersections.

    .. grid-item-card:: Mesh
        :link: mapper/mesh
        :link-type: doc

        Triangle mesh intersections (trimesh), complex surfaces.
