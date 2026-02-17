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
import numpy as np
import pyproj.network
from pyproj import CRS

from weitsicht import CameraOpenCVPerspective, ImagePerspective, MappingHorizontalPlane, Rotation

# PROJ activate network grid/data
pyproj.network.set_network_enabled(True)  # pyright: ignore[reportPrivateImportUsage]

# CRS
# Coordinate System we want to use
crs = CRS("EPSG:25833+3855")
# This will be UTM/Zone 33 North with vertical reference EGM2008

# MAPPER
mapper = MappingHorizontalPlane(plane_altitude=250.0, crs=crs)

# CAMERA
cam = CameraOpenCVPerspective(
    width=11648,
    height=8736,
    fx=12906.9238,
    fy=12906.9238,
    cx=5858.437,
    cy=4400.116,
    k1=-0.00565896,
    k2=0.0334872,
    k3=-0.0683847,
    p1=0.000933359,
    p2=0.00065228,
)

# IMAGE
image = ImagePerspective(
    width=11648,
    height=8736,
    camera=cam,
    crs=crs,
    position=np.array([601968.0, 5340368.0, 320.0]),  # in image CRS
    orientation=Rotation.from_opk_degree(omega=10, phi=10, kappa=90.0),
    mapper=mapper,
)

# PROJECT
pts_world = np.array([[601932.34, 5340348.43, 254.1], [601948.43, 5340359.13, 251.3], [601969.76, 5340392.42, 250.1]])

proj = image.project(pts_world, crs_s=image.crs)

print("\nProjection of 3D coordinates into image")
if proj.ok is True:
    pixels_in_frame = proj.pixels[proj.mask]
    print("Valid pixels: ", pixels_in_frame)
    if not np.all(proj.mask):
        print("Outside image: ", proj.pixels[~proj.mask])
else:
    print("Projecting coordinates did not work")
    print(proj.error)

# MAPPING
print("\nMapping image points")
result_mapping = image.map_points([[1000, 800], [2500, 1600]])
if result_mapping.ok is True:
    coords_3d = result_mapping.coordinates[result_mapping.mask]
    normals = result_mapping.normals[result_mapping.mask]
    gsd = result_mapping.gsd  # mean ground sampling distance in mapper CRS units
    gsd_per_point = (
        result_mapping.gsd_per_point[result_mapping.mask] if result_mapping.gsd_per_point is not None else None
    )
    print("coordinates mapped:", coords_3d)
    print("normals:", normals)
    print("gsd:", gsd)
    print("gsd_per_point:", gsd_per_point)

# CHECK
print("\n Checks:")
if not result_mapping.ok:
    print("Projection problems:", result_mapping.issues)
if proj.issues:
    print("Mapping warnings:", proj.issues)
