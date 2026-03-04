.. _example-0402:

===========================================
04-02 - Footprint from Metadata (EOR)
===========================================

The following example demonstrates a small workflow for a **single image**:

- extract metadata (from the image via *exiftool* or from the stored ``.json``),
- parse EOR (position/orientation) and show the effect of ``to_utm=True``,
- build an :class:`~weitsicht.image.perspective.ImagePerspective` from metadata,
- map the image's center point and footprint onto a horizontal plane.

.. note:: This example can be used to run with exiftool from phil harvey and the wrapper PyExiftool. Still the same meta-data is saved as ``.json`` within the directory so that exiftool is not needed to run the example


.. figure:: /_static/example_images/04_02.jpg
     :align: center
     :alt: Images of drone flight

     One image from a drone flight.

-----------------------
Import Modules
-----------------------
.. literalinclude:: ../../../../examples/04_02_meta_eor.py
   :language: python
   :start-after: # Importing
   :end-before:  # Extract Meta Data


-----------------------
Extract Meta Data
-----------------------
.. literalinclude:: ../../../../examples/04_02_meta_eor.py
   :language: python
   :start-after: # Extract Meta Data
   :end-before:  # PyProj

-----------------------
PyProj
-----------------------
.. literalinclude:: ../../../../examples/04_02_meta_eor.py
   :language: python
   :start-after: # PyProj
   :end-before:  # Initiate Mapper class


-----------------------
Initiate Mapper class
-----------------------
.. literalinclude:: ../../../../examples/04_02_meta_eor.py
   :language: python
   :start-after: # Initiate Mapper class
   :end-before:  # Meta-Data Parsing

-----------------------
Meta-Data Parsing
-----------------------
.. literalinclude:: ../../../../examples/04_02_meta_eor.py
   :language: python
   :start-after: # Meta-Data Parsing
   :end-before:  # to_utm

------------------------------
EOR from meta data and to_utm
------------------------------
.. literalinclude:: ../../../../examples/04_02_meta_eor.py
   :language: python
   :start-after: # to_utm
   :end-before:  # Build Image from metadata

---------------------------
Build Image from metadata
---------------------------
.. literalinclude:: ../../../../examples/04_02_meta_eor.py
   :language: python
   :start-after: # Build Image from metadata
   :end-before:  # Map footprint and center point

--------------------------------------
Map footprint and center point
--------------------------------------
.. literalinclude:: ../../../../examples/04_02_meta_eor.py
   :language: python
   :start-after: # Map footprint and center point
   :end-before:  # Map point

--------------------------------------
Map digitized point
--------------------------------------

.. figure:: /_static/example_images/04_02_point.jpg
     :align: center
     :alt: digitized point

     Digitized point

.. literalinclude:: ../../../../examples/04_02_meta_eor.py
   :language: python
   :start-after: # Map point


.. figure:: /_static/example_images/04_02_footprint.jpg
     :align: center
     :alt: Footprint over satelite imagery.

     Footprint and mapped point.
