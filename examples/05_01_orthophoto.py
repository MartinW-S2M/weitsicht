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
from pathlib import Path

import numpy as np
import pyproj.network
from pyproj import CRS

from weitsicht import ImageOrtho, MappingRaster

# Example data is stored under tests/data (shared with unit tests)
DATA_DIR = Path(__file__).parent.parent.resolve() / "tests" / "data"

# PyProj
# In that example we have different CRS systems of the image pose and the mapper
# and therefore we need to activate the network capabilities of pyproj to get the needed grids for transformation
pyproj.network.set_network_enabled(True)  # pyright: ignore[reportPrivateImportUsage]


# Image Class
# Load orthophoto from file
image = ImageOrtho.from_file(DATA_DIR / "44_2_op_2023_30cm.tif", crs=CRS("EPSG:31256+5778"))
if image is None:
    raise RuntimeError("Failed to load orthophoto")

# CRS is MGI Gauss Krueger M34
print("CRS: ", image.crs_wkt)
print(f"Resolution: {image.resolution:2.4f}")


# Mapper Class
# In this example we assign the mapper to the image after the image is already initialized.
# The DTM raster values are given in Vienna Heights (EPSG:8881)

# Load raster file for Mapper
image.mapper = MappingRaster(DATA_DIR / "44_2_dgm.tif", crs=CRS("EPSG:31256+8881"))

# Map center point
result_center = image.map_center_point()
if result_center.ok is True:
    print("\nMapped center point:", result_center.coordinates)
    print(f"GSD at center pixel: {result_center.gsd:2.3f}")
    print("Normal (center):", result_center.normals[result_center.mask])
    if result_center.gsd_per_point is not None:
        print("GSD per point:", result_center.gsd_per_point[result_center.mask])
else:
    raise RuntimeError(f"Mapping center point failed: {result_center.error} ({result_center.issues})")

# Map footprint
result_footprint = image.map_footprint()
if result_footprint.ok is True:
    print("\nMapped footprint (corners):", result_footprint.coordinates)
    print(f"Area: {result_footprint.area:2.0f}")
    print(f"GSD:  {result_footprint.gsd:2.3f}")
    print("Normals:", result_footprint.normals)
    if result_footprint.gsd_per_point is not None:
        print("GSD per point:", result_footprint.gsd_per_point)
else:
    raise RuntimeError(f"Mapping footprint failed: {result_footprint.error} ({result_footprint.issues})")

# Map densified footprint edges
result_footprint_dense = image.map_footprint(points_per_edge=5)
if result_footprint_dense.ok is True:
    print("\nMapped footprint (densified):", result_footprint_dense.coordinates)
    print("Number of points:", result_footprint_dense.coordinates.shape[0])
    print(f"Area: {result_footprint_dense.area:2.0f}")
    print(f"GSD:  {result_footprint_dense.gsd:2.3f}")
    print("Normals:", result_footprint_dense.normals)
    if result_footprint_dense.gsd_per_point is not None:
        print("GSD per point:", result_footprint_dense.gsd_per_point)
else:
    raise RuntimeError(
        f"Mapping densified footprint failed: {result_footprint_dense.error} ({result_footprint_dense.issues})"
    )

# Project + map_points roundtrip
# Coordinates are in the orthophoto CRS (x/y).
garden = np.array(
    [
        [-1563.025, 338413.602],
        [-1560.834, 338407.607],
        [-1529.935, 338396.308],
        [-1524.170, 338398.614],
        [-1512.756, 338429.859],
        [-1513.563, 338438.737],
        [-1518.636, 338447.038],
        [-1526.245, 338452.226],
        [-1535.700, 338453.725],
        [-1546.191, 338450.612],
        [-1552.763, 338443.694],
        [-1563.025, 338413.602],
    ]
)

garden_3d = np.column_stack((garden, np.zeros((garden.shape[0], 1), dtype=garden.dtype)))

result_proj = image.project(garden_3d, crs_s=image.crs)
if result_proj.ok is True:
    print("\nGarden projected into orthophoto pixels:", result_proj.pixels)
    if result_proj.issues:
        print("Projection issues:", result_proj.issues)
else:
    raise RuntimeError(f"Projecting garden coordinates failed: {result_proj.error} ({result_proj.issues})")

result_map = image.map_points(result_proj.pixels)
if result_map.ok is True:
    print("\nGarden mapped from projected pixels (with heights):", result_map.coordinates)
    print(f"Mean GSD: {result_map.gsd:2.3f}")
    print("Normals:", result_map.normals)
    if result_map.gsd_per_point is not None:
        print("GSD per point:", result_map.gsd_per_point)
else:
    raise RuntimeError(f"Mapping projected garden pixels failed: {result_map.error} ({result_map.issues})")

# Checks
np.testing.assert_almost_equal(result_map.coordinates[:, :2], garden[:, :2])
