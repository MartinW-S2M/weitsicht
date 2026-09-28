
.. currentmodule:: weitsicht

================
UTM Converter
================

Functions dealing with the transformation of coordinates an orientation to UTM/WGS84

Get UTM Zone
=============
Get the UTM zone for latitude and longitude given in degree

.. autofunction:: weitsicht.transform.utm_converter.get_zone

Convert coordinates to utm/wgs84 from any crs
=============================================
This transforms a point to UTM/WGS84 from any system via the rout to EPSG:4979
With the parameter "to_geoid_height" it can be specified that the result is either in ellipsoid heights or EGM2008

.. autofunction:: weitsicht.transform.utm_converter.point_convert_utm_wgs84

Convert coordinates to utm/wgs84 (EGM2008) from any crs
=======================================================
This transforms a point to UTM/WGS84 in the height EGM2008

.. autofunction:: weitsicht.transform.utm_converter.point_convert_utm_wgs84_egm2008

Convert coordinates to utm/wgs84 from wgs84ell
==============================================
This transforms a point to UTM/WGS84. The point needs to be given in WGS84 ellipsoidal coordinates
With the parameter "to_geoid_height" it can be specified that the result is either in ellipsoid heights or EGM2008

.. autofunction:: weitsicht.transform.utm_converter.point_wgs84ell_to_utm
