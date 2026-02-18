.. _example-0101:

==========================================
01-01 - Footprint & CenterPoint
==========================================

The following example illustrates how to calculate the image's footprint and center point.

To map image coordinates one need minimum the following information:

* Image EOR (exterior orientation: Position and Attitude of the projection center
* Image IOR (inner orientation: Focal length (the minimum to know), principal point and distortion parameters
* 3D information about the geometry which is captured by the image (Terrain Model, Mesh). In this example terrain data in form of a 1 meter resolution DEM is used from queensland.

For the following image the footprint and center point should be mapped:

.. image:: /_static/example_images/241214-161759-B50-0330_scr.jpg
  :width: 400
  :align: center
  :alt: Ariel image

|

Following Steps need to be done:

1. Create camera class (IOR of image)
2. Create image class (Containing EOR and camera)
3. Create mapper class (3D data for mapping)
4. Map footprint and center-point (Function calls of image class)

---------------
Import Modules
---------------
First we will import all necessary Modules. The Raster Mapper module will be used for mapping.

.. literalinclude:: ../../../../examples/01_01_map_footprint_centerpoint.py
   :language: python
   :start-after: # Importing
   :end-before:  # PyProj


------
PYPROJ
------
.. literalinclude:: ../../../../examples/01_01_map_footprint_centerpoint.py
   :language: python
   :start-after: # PyProj
   :end-before:  # Camera Model


------------
Camera Model
------------
.. literalinclude:: ../../../../examples/01_01_map_footprint_centerpoint.py
   :language: python
   :start-after: # Camera Model
   :end-before:  # Image Class

-----------
Image Class
-----------
.. literalinclude:: ../../../../examples/01_01_map_footprint_centerpoint.py
   :language: python
   :start-after: # Image Class
   :end-before:  # Mapper Class

------------
Mapper Class
------------
.. literalinclude:: ../../../../examples/01_01_map_footprint_centerpoint.py
   :language: python
   :start-after: # Mapper Class
   :end-before:  # Results Image 1


-------------------------------------
Map images footprint and center point
-------------------------------------
.. literalinclude:: ../../../../examples/01_01_map_footprint_centerpoint.py
   :language: python
   :start-after: # Results Image 1
   :end-before:  # Map densified footprint edges


The mapped footprint and center point of that image (red dots are the mapped points of the footprint):

.. image:: /_static/example_images/footprint_01.jpg
  :width: 400
  :align: center
  :alt: Alternative text

-------------------------
Densify mapped footprint
-------------------------
.. literalinclude:: ../../../../examples/01_01_map_footprint_centerpoint.py
   :language: python
   :start-after:  # Map densified footprint edges

Now there are 3 additional points on each edge (red dots):

.. image:: /_static/example_images/footprint_02.jpg
  :width: 400
  :align: center
  :alt: Alternative text
