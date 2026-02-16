.. _example-0301:

===========================
03-01 - Footprint of Images
===========================

The following example illustrates how to use and initialize image batches.
This can be useful if you want to run methods on a batch of images.

In  this case, a ariel flight with 2 cameras pointing oblique right and left away.
for the cameras, pre-calibration exists and is used for the camera class.

The image orientation are stored in a CSV file and is directly derived from a IMU postprocessing.
This is a good case example as for that data projection and mapping will not be as accurate if derived by a proper photogrammetric block.
But special for such data this tool can provide an easy way to get footprint, mapped areas or approximate image pixel location for projected 3D points.

The Raster for mapping is a cutout from an austrian DTM with a resolution of 10 meter.
The coordinate system of the raster is austrian lambert projection with the vertical datum GHA (austrian heights in use).

The workflow and methods are not much different to single images.

Example images:

.. image:: /_static/example_images/batch_overview.jpg
  :width: 400
  :align: center
  :alt: Image batch


Following Steps need to be done:
1. Create mapper class
2. Create camera class
3. Create image class
4. Create ImageBatch class
5. Map footprint


---------------
Import Modules
---------------
First we will import all needed Modules. The Raster Mapper module will be used for mapping.

.. literalinclude:: ../../../../examples/03_01_image_batch.py
   :language: python
   :start-after: # Importing
   :end-before:  # Mapper Class

-------------
Mapper Class
-------------
.. literalinclude:: ../../../../examples/03_01_image_batch.py
   :language: python
   :start-after: # Mapper Class
   :end-before:  # Camera Model


---------------
Camera Model
---------------
Initialize the camera models of the image

.. literalinclude:: ../../../../examples/03_01_image_batch.py
   :language: python
   :start-after: # Camera Model
   :end-before:  # Parse images from CSV

----------------------
Parse images from CSV
----------------------

.. literalinclude:: ../../../../examples/03_01_image_batch.py
   :language: python
   :start-after: # Parse images from CSV
   :end-before:  # ImageBatch Class

----------------
ImageBatch Class
----------------
.. literalinclude:: ../../../../examples/03_01_image_batch.py
   :language: python
   :start-after: # ImageBatch Class
   :end-before:  # Map all footprints

------------------------------------
Map images footprints of image batch
------------------------------------
.. literalinclude:: ../../../../examples/03_01_image_batch.py
   :language: python
   :start-after: # Map all footprints

The mapped footprints of the images of the batch:

.. image:: /_static/example_images/image_batch_footprints.jpg
  :width: 400
  :align: center
  :alt: Footprints of image batch

------------------------------------
Map images footprints of image batch
------------------------------------
