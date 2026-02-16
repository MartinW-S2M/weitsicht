.. _example-0302:

==============================================
03-02 - Project 3D Coordinates
==============================================

The following example shows how to use the project command for image batches.
It will calculate for all images the projection of coordinates and return if wanted only the one with valid projections.


Here we use the same image batch as already created in example :doc:`03-01 - Footprint of Images <03_01_image_batch>`



.. literalinclude:: ../../../../examples/03_02_image_batch_project.py
   :language: python
   :end-before:  # Map Polygon

---------------------------------------------
Map Points
---------------------------------------------
The following line should be mapped. The mapped coordinates are used to find projections in images.

.. image:: /_static/example_images/ariel_coordinates_digitized_P0009912.jpg
  :width: 400
  :align: center
  :alt: Projected coordinates


.. literalinclude:: ../../../../examples/03_02_image_batch_project.py
   :language: python
   :start-after: # Map Polygon
   :end-before:  # Project coordinates on images

---------------------------------------------
Project coordinates on images
---------------------------------------------
.. literalinclude:: ../../../../examples/03_02_image_batch_project.py
   :language: python
   :start-after: # Project coordinates on images

The projected coordinates for two images P0009077, P0009093.

.. note:: That examples exterior orientation is directly logged from an INS(IMU/GNSS). No bundle adjustment was performed to optimize EOR or IOR (from pre-calibration)


.. image:: /_static/example_images/ariel_project.jpg
  :width: 400
  :align: center
  :alt: Projected coordinates
