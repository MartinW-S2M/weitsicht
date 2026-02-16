:notoc:

.. module:: weitsicht

.. raw:: html

   <img src="_static/weitsicht.svg" alt="weitsicht logo" style="width:150px; float:right; margin:0 0 0 12px;" />


************************
weitsicht documentation
************************

**Date**: |today| **Version**: |version|


**Useful links**:
`Source Repository <https://github.com/MartinW-S2M/weitsicht>`__ |
`Issues & Ideas <https://github.com/MartinW-S2M/weitsicht/issues>`__ |
`Q&A Support <https://github.com/MartinW-S2M/weitsicht/discussions>`__ |


.. admonition:: Unlock Your Geo-Referenced Images
   :class: tip

   | **You have geo-referenced images but do not know how to further use them?**
   | **Great, exactly here weitsicht jumps in. Use the full potential of your geo-referenced images and map your pixels or project objects into the image.**
   | What are geo-referenced images: :ref:`geo-ref-basic`

``weitsicht`` is an open source, Apache 2.0 library  for `Python <https://www.python.org/>`__.
The library allows to work with geo-referenced image data. Image pixel coordinates can be mapped to 3D or 3D points can be projected
into images to get pixel coordinates.


**The package bridges computer-vision/photogrammetry outputs and GIS workflows, carrying data from direct or indirect georeferencing into downstream uses like monitoring and digitization.**

.. list-table::
   :width: 80%
   :class: borderless

   * - .. image:: /_static/raster_mapping.jpg
          :width: 80%

     - .. image:: /_static/mesh_pic.jpg
          :width: 80%

     - .. image::  /_static/example_images/image_batch_footprints.jpg
          :width: 100%


Capabilities:

- **Mapping**, map the image's center-point and footprint (image extent) easily.
- **Projection**, get the pixel position of 3D coordinates.
- **CRS**, weitsicht handles coordinate system conversions (to some extent)
- **Perspective Image and Camera**, mathematic model of your digital camera and pose.
- **Ortho imagery**, use ortho imagery to map content or convert 2D coordinates to 3D.
- **Mapper Classes**, several mapper classes can be used to map your pixel data: HorizontalPlane, Raster, Mesh
- **ImageBatch**, container class to perform tasks on multiple images. Find all images where coordinates are visible. Map for all images footprint and centerpoint.
- **Meta-Data**, use image's meta-data (EXIF, XMP) to estimate camera model and image pose.

``weitsicht`` is developed to provide an easy to use package for all levels of experience,
from scientist in environmental science without computer vision or photogrammetric background,
for teaching and students as well for computer vision/photogrammetry experts.

Its structure is kept as modular as possible to easily extend new mathematical models for cameras, images, or mappers.

``weitsicht`` is not a Structure from Motion (SFM) package.

.. note::
    As of the nature of python, do not expect to perform super fast.
    Especially the processing intensive operations on raster and mesh.
    Nevertheless some effort was put into optimization and reduction of loops.



.. grid:: 1 2 2 2
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card::  User Guide
        :img-top: _static/index_getting_started.svg
        :link: user_guides/index
        :link-type: any
        :class-card: intro-card
        :shadow: md

        The user guide provides information about installation and how to use the package.

        +++


    .. grid-item-card:: Documentation
        :img-top: _static/index_user_guide.svg
        :link: documentation/index
        :link-type: any
        :class-card: intro-card
        :shadow: md

        The documentation provides background, definitions, and mathematical context.

        +++


    .. grid-item-card::  API reference
        :img-top: _static/index_api.svg
        :link: api/index
        :link-type: any
        :class-card: intro-card
        :shadow: md

        The reference guide contains a detailed description of
        the API. The reference describes how the methods work and which parameters can
        be used. It assumes that you have an understanding of the key concepts.

        +++

    .. grid-item-card::  Contribution
        :img-top: _static/index_contribute.svg
        :link: contribution
        :link-type: any
        :class-card: intro-card
        :shadow: md

        Contributions are highly welcome. There are many ways to contribute.
        Fixing typos, extending functionalities, implementing new classes...

        +++


------------------------
WISDAMapp
------------------------
WISDAMapp is the software packages which provides a GUI to work with environmental images.
Images can be loaded and digitized, for example to monitor marine mammals. There are workflows for metadata enrichment as well as workflows to work with AI detections.

WISDAMapp was one of the main roots for that package. Originally it was implemented completely inside WISDAMapp.

As there was interest in the mathematical/geometrical core behind WISDAMapp, the first major refactoring of WISDAMapp in late 2024 was the right time to split the GUI and the core into two parts.

.. list-table::
   :header-rows: 0
   :widths: 60 40
   :align: center
   :class: borderless

   * - **Project website:** `www.wisdamapp.org <https://www.wisdamapp.org>`__

       **Repository:** `github.com/WISDAMapp/WISDAM <https://github.com/WISDAMapp/WISDAM>`__
     - .. image:: _static/WISDAM_Hero_Logo_Black.svg
          :width: 200
          :align: center
          :alt: WISDAMapp logo


.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:

    user_guides/index
    documentation/index
    examples/index
    api/index
    contribution
    change_log


------------------------
Examples
------------------------

.. grid:: 2 2 4 4
    :gutter: 1

    .. grid-item-card:: Footprint
        :img-bottom: _static/example_images/footprint_02.jpg
        :link: example-0101
        :link-type: ref
        :shadow: md


    .. grid-item-card::  Map image points
        :img-bottom: _static/example_images/tennis_court.jpg
        :link: example-0102
        :link-type: ref
        :shadow: md


    .. grid-item-card::  Project 3D points
        :img-bottom: _static/example_images/pixel_image1_graffito_text.jpg
        :link: example-0201
        :link-type: ref
        :shadow: md

    .. grid-item-card::  Footprints with image batch
        :img-bottom: _static/example_images/image_batch_footprints.jpg
        :link: example-0301
        :link-type: ref
        :shadow: md

    .. grid-item-card::  Projections with image batch
        :img-bottom: _static/example_images/batch_overview.jpg
        :link: example-0302
        :link-type: ref
        :shadow: md

    .. grid-item-card::  Full workflow - Survey digitization
        :img-bottom: _static/dugong.jpg
        :link: example-0401
        :link-type: ref
        :shadow: md

    .. grid-item-card::  Orthophoto
        :img-bottom: /_static/example_images/ortho_vienna.jpg
        :link: example-0501
        :link-type: ref
        :shadow: md
