.. _example-0103:

========================
01-03 - Mesh Mapper
========================

The following example illustrates how to calculate the image's footprint and center point using a mesh as geometry reference.
The mapping class for meshes is called MappingTrimesh as ``Trimesh`` is used as the python library behind.

.. image:: /_static/example_images/mesh_graffito.JPG
  :width: 70%
  :align: center
  :alt: Image of graffiti

For the following images the footprint and center point should be mapped:

.. image:: /_static/example_images/img_portrait_graffiti_wall.jpg
  :width: 40%
  :align: center
  :alt: Image of graffiti


.. image:: /_static/example_images/image_larger_footprint.jpg
  :width: 60%
  :align: center
  :alt: Image of graffiti

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

.. literalinclude:: ../../../../examples/01_03_mesh_example.py
   :language: python
   :start-after: # Importing
   :end-before:  # Coordinate System


--------------------
Coordinate System
--------------------
In that example the images EOR and the mappers share the same coordinate system, thus we will not use crs specifications.

.. literalinclude:: ../../../../examples/01_03_mesh_example.py
   :language: python
   :start-after: # Coordinate System
   :end-before:  # Camera Class

---------------
Camera Model
---------------
In that example the images EOR and the mappers share the same coordinate system, thus we will not use crs specifications.

.. literalinclude:: ../../../../examples/01_03_mesh_example.py
   :language: python
   :start-after: # Camera Class
   :end-before:  # Mapper Class

---------------
Mapper Class
---------------
.. literalinclude:: ../../../../examples/01_03_mesh_example.py
   :language: python
   :start-after: # Mapper Class
   :end-before:  # Image Class

---------------
Image Class
---------------
.. literalinclude:: ../../../../examples/01_03_mesh_example.py
   :language: python
   :start-after: # Image Class
   :end-before:  # Map images footprint and center point


Map images footprint and center point
----------------------------------------
.. literalinclude:: ../../../../examples/01_03_mesh_example.py
   :language: python
   :start-after: # Map images footprint and center point
   :end-before:  # Densify mapped footprint

The mapped footprint and center point of that image (red dots are the mapped points of the footprint):

.. image:: /_static/example_images/footprint_mesh.jpg
  :width: 400
  :align: center
  :alt: Alternative text

-------------------------
Densify mapped footprint
-------------------------
Standard is that only the four corner points of the image are mapped.
If you provide the argument ``points_per_edge`` you can specify how many points per side should be added.
This can be useful if the footprint outline should be more accurate on non-flat terrain or large camera distortions

.. literalinclude:: ../../../../examples/01_03_mesh_example.py
   :language: python
   :start-after: # Densify mapped footprint
   :end-before:  # Second image


Now there are 3 additional points on each edge (red dots):

.. image:: /_static/example_images/footprint_mesh_02.jpg
  :width: 400
  :align: center
  :alt: Alternative text


---------------
Second Image
---------------

We do the same for another image of the same photogrammetric block.
It is the same camera used which shares the same camera calibration, therefore we can reuse cam for ``image_2``
That image has a larger footprint as the geometric extend of the mesh model, so this footprint mapping will fail.

.. literalinclude:: ../../../../examples/01_03_mesh_example.py
   :language: python
   :start-after: # Second image
