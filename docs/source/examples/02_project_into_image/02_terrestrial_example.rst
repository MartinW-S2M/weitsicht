.. _example-0201:

===========================
02-01 - Terrestrial Example
===========================

The following example illustrates how you can project 3D points back into the image.

We want to project the digitized 3D outline of a graffito painted on a wall. The points have been digitized using the Mesh and Texture.
The 3D model was created by photogrammetry where as result we get the image exterior orientation and the camera calibration.

.. image:: /_static/example_images/mesh_graffito_3d_points.jpg
  :width: 400
  :align: center
  :alt: Image of Mesh with digitized Points of outline of one graffito

The green line shows the digitized points. The order of the points will be preserved in the output.

Following Steps need to be done:

1. Create camera class
2. Create image class
3. Project 3D points into image
4. Check validity of projected points

---------------
Import Modules
---------------
First we will import all needed Modules

.. literalinclude:: ../../../../examples/02_01_project_points.py
   :language: python
   :start-after: # Importing
   :end-before:  # 3D points

----------
3D points
----------
.. literalinclude:: ../../../../examples/02_01_project_points.py
   :language: python
   :start-after: # 3D points
   :end-before:  # Camera Model

---------------
Camera Model
---------------
.. literalinclude:: ../../../../examples/02_01_project_points.py
   :language: python
   :start-after: # Camera Model
   :end-before:  # Image Class

---------------
Image Class
---------------
Next we use the camera and the image information to initialize our image class.
To have a geo-referenced image we need at least know the exterior orientation: Position and Attitude/Orientation

.. literalinclude:: ../../../../examples/02_01_project_points.py
   :language: python
   :start-after: # Image Class
   :end-before:  # Calculate Projections

---------------------
Calculate Projections
---------------------
.. literalinclude:: ../../../../examples/02_01_project_points.py
   :language: python
   :start-after: # Calculate Projections
   :end-before:  # Second Image

The pixel position of the projected points look like this (light green line):

.. image:: /_static/example_images/pixel_image1_graffito_text.jpg
  :width: 400
  :align: center
  :alt: Alternative text

---------------
Second Image
---------------

We do the same for another image of the same photogrammetric block.
It is the same camera used which shares the same camera calibration, therefore we can reuse cam for ``image_2``

.. literalinclude:: ../../../../examples/02_01_project_points.py
   :language: python
   :start-after: # Second Image


The pixel position of the projected points look like this for the second image (light green line):

.. image:: /_static/example_images/pixel_image2_graffito_text.jpg
  :width: 400
  :align: center
  :alt: Alternative text
