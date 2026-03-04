# -----------------------------------------------------------------------
# Copyright 2026 Martin Wieser
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------

# Importing
import json
from pathlib import Path

import numpy as np
from pyproj import CRS
from pyproj import network as pyproj_network

from weitsicht import (
    CoordinateTransformer,
    MappingHorizontalPlane,
    PyExifToolTags,
    eor_from_meta,
    image_from_meta,
)
from weitsicht.geometry.coo_geojson import get_geojson

DATA_DIR = Path(__file__).parent.resolve() / "data"
IMAGE_PATH = DATA_DIR / "DSC00316.jpg"
META_PATH = DATA_DIR / "DSC00316.json"

# Extract Meta Data
# This example can be used to run with exiftool from phil harvey.
# Still the same meta-data is saved under within the directory so that exiftool is not needed to run the example

# To run with pyexiftool under windows you will need to specify the exiftool.exe path either in your PATHS
# or specify the path to the executable manually --> et = ExifToolHelper(executable=PATH_TO_EXIF)
et_helper = None
try:
    from exiftool import ExifToolHelper  # pyright: ignore[reportMissingImports]

    try:
        et_helper = ExifToolHelper()
        # You can specify the exif executible via ExifToolHelper((executable=r"..\exiftoll.exe")
        print("Exiftool via PyExifTool is used")
    except (RuntimeError, TypeError, NameError, FileNotFoundError):
        pass
except ModuleNotFoundError:
    pass

if et_helper is not None and IMAGE_PATH.is_file():
    meta_data = et_helper.get_tags(IMAGE_PATH, tags=None)[0]
else:
    meta_data = json.load(open(META_PATH))


# PyProj
# In that example we will convert the pose to UTM (``to_utm=True``) which yields a compound CRS (UTM + EGM2008, +3855).
# This can require PROJ grid data. Enable pyproj's network mode to let PROJ download missing grids when needed.
pyproj_network.set_network_enabled(True)  # type: ignore

# Initiate Mapper class
# For that example we map the footprint and center point on a horizontal plane (e.g. sea level with 0.0m).
# The mapper uses WGS84 horizontal coordinates and EGM2008 orthometric heights.
crs_mapper = CRS("EPSG:4326+3855")
mapper_0h = MappingHorizontalPlane(plane_altitude=0.0, crs=crs_mapper)

# Meta-Data Parsing
tag_loader = PyExifToolTags(meta_data)
tags = tag_loader.get_all()
# You can check beforehand if the tags contain crs for example:
# This example contains information about both horizontal and vertical cs

print("Horizontal CS: ", tags.crs.horiz_cs)
print("Vertical CS: ", tags.crs.vert_cs)
print("\n")

# to_utm
# eor_from_meta returns the pose in WGS84 ECEF (EPSG:4978) by default.
eor_res = eor_from_meta(tags)
if eor_res.ok is False:
    raise RuntimeError(f"Failed to extract EOR from metadata: {eor_res.error} ({eor_res.issues})")
print("EOR (default) CRS:", eor_res.crs)
print("EOR (default) position:", eor_res.position)

# With to_utm=True, the pose is converted to a local UTM compound CRS (UTM + EGM2008 height, +3855).
eor_res_utm = eor_from_meta(tags, to_utm=True)
if eor_res_utm.ok is False:
    raise RuntimeError(f"Failed to extract EOR (UTM) from metadata: {eor_res_utm.error} ({eor_res_utm.issues})")
print("EOR (to_utm=True) CRS:", eor_res_utm.crs)
print("EOR (to_utm=True) position:", eor_res_utm.position)

# Build Image from metadata
img_res = image_from_meta(tags)  # returns an ImagePerspective with ECEF pose
if img_res.ok is False:
    raise RuntimeError(f"Failed to build image from metadata: {img_res.error} ({img_res.issues})")
image_ecef = img_res.image

img_res_utm = image_from_meta(tags, to_utm=True)  # same image but pose in UTM (+3855)
if img_res_utm.ok is False:
    raise RuntimeError(f"Failed to build image (UTM) from metadata: {img_res_utm.error} ({img_res_utm.issues})")
image_utm = img_res_utm.image

# Map footprint and center point
# When mapping, the resulting coordinates are returned in the CRS of the image.
center_ecef = image_ecef.map_center_point(mapper=mapper_0h)
assert center_ecef.ok is True  # for testing

center_utm = image_utm.map_center_point(mapper=mapper_0h)
assert center_utm.ok is True  # for testing

footprint_utm = image_utm.map_footprint(mapper=mapper_0h)
assert footprint_utm.ok is True  # for testing

print("Mapped centerpoint (ECEF):", center_ecef.coordinates[center_ecef.mask])
print("Mapped centerpoint (UTM):", center_utm.coordinates[center_utm.mask])
print("Mapped footprint (UTM):", footprint_utm.coordinates)
print(f"Footprint area (UTM): {footprint_utm.area:2.0f} m²")

# Verify that to_utm is consistent by transforming the mapped centerpoint from ECEF -> UTM.
ct_ecef_to_utm = CoordinateTransformer.from_crs(center_ecef.crs, center_utm.crs)
assert ct_ecef_to_utm is not None  # for testing
center_ecef_as_utm = ct_ecef_to_utm.transform(center_ecef.coordinates)
assert np.allclose(center_ecef_as_utm, center_utm.coordinates, atol=0.05)  # for testing

# geojson
# We can use a little helper function to get geojson geometry
geojson = get_geojson(coordinates=footprint_utm.coordinates, geom_type="Polygon")
print(geojson)

# This we could use to import to QGIS using QuickGEOJSON

# Map point
# We now map a digitized point with pixel coordiantes 2330,1890
point_utm = image_utm.map_points(np.array([[2330, 1890]]), mapper=mapper_0h)
if point_utm.ok is True:
    print("PIONT coordinate:", point_utm.coordinates)
