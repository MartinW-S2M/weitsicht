.. _example-0501:

===========================
05-01 - OrthoImagery
===========================

The following example demonstrates how to work with geo-referenced orthophoto imagery:

- map the center point (samples height via a mapper),
- map the footprint (corners and densified edges),
- project world coordinates into orthophoto pixels and map them back to 3D.

.. image:: /_static/example_images/ortho_vienna.jpg
   :width: 400
   :align: center
   :alt: Orthophoto example (Vienna)


---------------
Import Modules
---------------
.. literalinclude:: ../../../../examples/05_01_orthophoto.py
   :language: python
   :start-after: # Importing
   :end-before: # PyProj


------
PYPROJ
------
.. literalinclude:: ../../../../examples/05_01_orthophoto.py
   :language: python
   :start-after: # PyProj
   :end-before: # Image Class


---------------------------------
Initialize Image class
---------------------------------
.. literalinclude:: ../../../../examples/05_01_orthophoto.py
   :language: python
   :start-after: # Image Class
   :end-before: # Mapper Class


---------------------------------
Initialize Mapper class
---------------------------------
.. literalinclude:: ../../../../examples/05_01_orthophoto.py
   :language: python
   :start-after: # Mapper Class
   :end-before: # Map center point


----------------
Map center point
----------------
.. literalinclude:: ../../../../examples/05_01_orthophoto.py
   :language: python
   :start-after: # Map center point
   :end-before: # Map footprint


-------------
Map footprint
-------------
.. literalinclude:: ../../../../examples/05_01_orthophoto.py
   :language: python
   :start-after: # Map footprint
   :end-before: # Map densified footprint edges


-----------------------
Densify footprint edges
-----------------------
.. literalinclude:: ../../../../examples/05_01_orthophoto.py
   :language: python
   :start-after: # Map densified footprint edges
   :end-before: # Project + map_points roundtrip


----------------------------------
Project and map points roundtrip
----------------------------------
.. literalinclude:: ../../../../examples/05_01_orthophoto.py
   :language: python
   :start-after: # Project + map_points roundtrip
   :end-before: # Checks


------
Checks
------
.. literalinclude:: ../../../../examples/05_01_orthophoto.py
   :language: python
   :start-after: # Checks
