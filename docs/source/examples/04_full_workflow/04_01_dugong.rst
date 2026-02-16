.. _example-0401:

===========================================
04-01 - Digitize Dugongs
===========================================

The following example demonstrates a full workflow where a set of images is stored in a folder (e.g. drone flight)
and how digitized objects, a dugong in this case, can be mapped and the approximate appearance in other images can be estimated.


.. figure:: /_static/example_images/monitoring_sample.jpg
     :align: center
     :alt: Images of drone flight

     Sample images of a drone flight.

.. note:: This example can be used to run with exiftool from phil harvey and the wrapper PyExiftool. Still the same meta-data is saved as ".json" within the directory so that exiftool is not needed to run the example

-----------------------
Import Modules
-----------------------
.. literalinclude:: ../../../../examples/04_01_dugong_survey.py
   :language: python
   :start-after: # Importing
   :end-before:  # Extract Meta Data


-----------------------
Extract Meta Data
-----------------------
.. literalinclude:: ../../../../examples/04_01_dugong_survey.py
   :language: python
   :start-after: # Extract Meta Data
   :end-before:  # PyProj

-----------------------
PyProj
-----------------------
.. literalinclude:: ../../../../examples/04_01_dugong_survey.py
   :language: python
   :start-after: # PyProj
   :end-before:  # Initiate Mapper class


-----------------------
Initiate Mapper class
-----------------------
.. literalinclude:: ../../../../examples/04_01_dugong_survey.py
   :language: python
   :start-after: # Initiate Mapper class
   :end-before:  # Meta-Data Parsing

-----------------------
Meta-Data Parsing
-----------------------
.. literalinclude:: ../../../../examples/04_01_dugong_survey.py
   :language: python
   :start-after: # Meta-Data Parsing
   :end-before:  # Image Batch

-----------------------
Image Batch
-----------------------
.. literalinclude:: ../../../../examples/04_01_dugong_survey.py
   :language: python
   :start-after: # Image Batch
   :end-before:  # Mapping of point

--------------------------
Mapping of digitized point
--------------------------
In that example a dugong was found in one of the images and digitized by hand or by AI.
The center of the objects which was digitized in `image003.jpg` is:

    1292, 564

.. figure:: /_static/example_images/dugong_point_digitized_img003.jpg
     :align: center
     :alt: Digitized dugong

     The digitized dugong in image 003

.. literalinclude:: ../../../../examples/04_01_dugong_survey.py
   :language: python
   :start-after: # Mapping of point
   :end-before:  # Find valid projections

--------------------------------------
Find projection in all other images
--------------------------------------

.. literalinclude:: ../../../../examples/04_01_dugong_survey.py
   :language: python
   :start-after: # Find valid projections
   :end-before:  # Filtered Result

No we get for all images in the image batch the projected pixels and a mask if the projection is valid.
A projection is only valid if its inside the distortion border of the image.

image001 (array([[1385.37313612,   51.86895822]]), array([ True]))
image002 (array([[1320.55732947,  337.27117976]]), array([ True]))
image003 (array([[1292.,  564.]]), array([ True]))
image004 (array([[1111.88306103,  929.0284043 ]]), array([ True]))

.. figure:: /_static/example_images/projections_04_01.jpg
     :align: center
     :alt: Images of drone flight




This information can than be used to find the objects in other images in a GUI like **WISDAMapp** easier.

--------------------------------------
Filtered Result
--------------------------------------
.. literalinclude:: ../../../../examples/04_01_dugong_survey.py
   :language: python
   :start-after: # Filtered Result

Filtered projections. Only one image is returned

image004 (array([[1413.14434847,  968.62647658]]), array([ True]))
