
.. currentmodule:: weitsicht

================
UTM Converter
================

Functions dealing with the transformation of coordinates an orientation to UTM/WGS84

Get UTM Zone
=============
Get the UTM zone for latitued and longitude given in degree

.. autofunction:: weitsicht.transform.utm_converter.get_zone

Convert coordinates to utm/wgs84 from any crs
=============================================
This transforms a point to UTM/WGS84 in the height EGM2008

.. autofunction:: weitsicht.transform.utm_converter.point_convert_utm_wgs84_egm2008

Convert coordinates to utm/wgs84 from wgs84ell
==============================================
This transforms a point to UTM/WGS84 in the height EGM2008

.. autofunction:: weitsicht.transform.utm_converter.point_wgs84ell_to_utm
