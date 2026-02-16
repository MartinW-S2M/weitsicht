:html_theme.sidebar_secondary.remove: true


.. toctree::
   :maxdepth: 2
   :caption: Mapping/Monoplotting
   :hidden:
   :glob:

   01_mapping/*

.. toctree::
   :maxdepth: 1
   :caption: Project coordinates
   :hidden:
   :glob:

   02_project_into_image/*


.. toctree::
   :maxdepth: 1
   :caption: Batch of images
   :hidden:
   :glob:

   03_batch_images/*

.. toctree::
   :maxdepth: 1
   :caption: Ortho Imagery
   :hidden:
   :glob:

   05_orthophoto/*


.. toctree::
   :maxdepth: 1
   :caption: Full workflow
   :hidden:
   :glob:

   04_full_workflow/*



==================
Examples
==================
Here you will find some examples how to use the package. The examples show how you can use the classes directly or use
the implemented helper workflows which try to estimate the needed information from image metadata.


Use the classes directly
========================
These example show how to use the classes for images and mapping directly,
for example if you have all the information needed from a service provider, a logfile or a SFM package.

All examples can be found under "examples" in the root directory of that repo.

.. grid:: 2 2 4 4
    :gutter: 1

    .. grid-item-card:: Footprint & Centerpoint
        :img-bottom: /_static/example_images/footprint_02.jpg
        :link: example-0101
        :link-type: ref
        :shadow: md


    .. grid-item-card::  Map image points
        :img-bottom: /_static/example_images/tennis_court.jpg
        :link: example-0102
        :link-type: ref
        :shadow: md

    .. grid-item-card::  Map on Mesh
        :img-bottom: /_static/example_images/footprint_mesh.JPG
        :link: example-0103
        :link-type: ref
        :shadow: md


    .. grid-item-card::  Project 3D points
        :img-bottom: /_static/example_images/pixel_image1_graffito_text.jpg
        :link: example-0201
        :link-type: ref
        :shadow: md

    .. grid-item-card::  Footprints with image batch
        :img-bottom: /_static/example_images/image_batch_footprints.jpg
        :link: example-0301
        :link-type: ref
        :shadow: md

    .. grid-item-card::  Projections with image batch
        :img-bottom: /_static/example_images/ariel_project.jpg
        :link: example-0302
        :link-type: ref
        :shadow: md

    .. grid-item-card::  Orthophoto
        :img-bottom: /_static/example_images/ortho_vienna.jpg
        :link: example-0501
        :link-type: ref
        :shadow: md


Helper Workflow
===========================
These example show how to use the classes for images and mapping directly,
for example if you have all the information needed from a service provider, a logfile or a SFM package.


.. grid:: 2 2 4 4
    :gutter: 1


    .. grid-item-card::  Full workflow - Survey digitization
        :img-bottom: /_static/dugong.jpg
        :link: example-0401
        :link-type: ref
        :shadow: md
