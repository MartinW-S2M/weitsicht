.. _example-0102:

==========================================
01-02 - Image Points
==========================================

The following example illustrates how to map image pixel coordinates.
Here we use the same data as already created in example :ref:`example-0101`

To map image coordinates one need minimum the following information:

* Image EOR (exterior orientation: Position and Attitude of the projection center
* Image IOR (inner orientation: Focal length (the minimum to know), principal point and distortion parameters
* 3D information about the geometry which is captured by the image (Terrain Model, Mesh). In this example terrain data in form of a 1 meter resolution DEM is used from queensland.

The digitized polygon on the following image should be mapped:

.. image:: /_static/example_images/241214-161759-B50-0330_scr.jpg
  :width: 400
  :align: center
  :alt: Ariel image

---------------
Preparation
---------------

.. literalinclude:: ../../../../examples/01_02_map_points.py
   :language: python
   :start-after: # Importing
   :end-before: # Map images points


-------------------------------------
Map images points
-------------------------------------
To map image points the image coordinates need to be specified which should be mapped.
The image shows the zoomed area of the tennis court with the digitized polygon in the image.

.. image:: /_static/example_images/tennis_court.jpg
  :width: 400
  :align: center
  :alt: Digitized tennis court

.. literalinclude:: ../../../../examples/01_02_map_points.py
   :language: python
   :start-after: # Map images points

..
  .. program-output:: python 01_02_map_points.py
   :cwd: ../../../../examples

  .. program-output:: python -c "import sys; print(sys.executable); print(sys.path[:5])"
   :cwd: ../../../../examples


The mapped points of the tennis court on a map.
**The returned coordinates will be in the image's crs**

.. image:: /_static/example_images/tennis_court_mapped.jpg
  :width: 400
  :align: center
  :alt: Mapped outline of tennis court
