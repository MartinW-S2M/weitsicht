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
import csv
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from pyproj import CRS
from pyproj import network as pyproj_network

from weitsicht import (
    CameraOpenCVPerspective,
    ImageBase,
    ImageBatch,
    ImagePerspective,
    MappingRaster,
    Rotation,
)

DATA_DIR = Path(__file__).parent.resolve() / "data" / "ariel_flight"

# Mapper Class
# In that example we have different CRS systems of the image pose and the mapper
# therefore we need to activate the network capabilities of pyproj to get the needed grids for transformation
pyproj_network.set_network_enabled(True)  # type: ignore

# Austrian Lambert with Vertical Datum GHA (Austrian Heights)
crs_mapper = CRS("EPSG:31287+5778")
mapper_raster = MappingRaster(
    raster_path=DATA_DIR / "dtm_epsg31287plus5778.tif",
    crs=crs_mapper,
    preload_full_raster=True,
)


# Camera Model
# Cameras from pre calibration protocol
camera_right = CameraOpenCVPerspective(
    width=11664,
    height=8750,
    fx=9524.786,
    fy=9524.788,
    cx=5811.896,
    cy=4422.312,
    k1=1.598e-02,
    k2=-7.004e-02,
    k3=1.940e-02,
    k4=6.022e-03,
    p1=1.016e-05,
    p2=1.284e-05,
)

camera_left = CameraOpenCVPerspective(
    width=11664,
    height=8750,
    fx=9520.292,
    fy=9520.292,
    cx=5850.933,
    cy=4375.201,
    k1=1.523e-02,
    k2=-6.958e-02,
    k3=1.942e-02,
    k4=5.851e-03,
    p1=-8.637e-05,
    p2=3.504e-05,
)

# Parse images from CSV
# The image ext ori is exported from the IMU/GNSS in UTM33 with ellipsoid heights.
# So to upgrade UTM to a 3D system using ellipsoid heights there is the command to_3d()
image_crs = CRS("EPSG:25833").to_3d()

# so type checker are not complaining that we are assigning ImageBase children to image_dict
image_dict: Mapping[str, ImageBase] = {}

# Parse the information of the exterior orientation from CSV files
# There are two files for the 2 cameras with the EOR information

# Using CSV reader as its part of python
# The csv files look like that following
# Filename,Easting,Northing,Height,Omega[deg],Phi[deg],Kappa[deg]
# P0009080_r1_cam3.jpg,604524.2107,5313148.5753,397.1245,20.350469123,-0.742671787,81.475314318
# P0009081_r1_cam3.jpg,604566.2565,5313155.3605,397.7679,19.673615822,0.774277423,84.003988648

# Images from the right camera
with open(DATA_DIR / "eor_camera_right.csv", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        pos = np.array([float(row["Easting"]), float(row["Northing"]), float(row["Height"])])
        rot = Rotation.from_opk_degree(
            omega=float(row["Omega[deg]"]),
            phi=float(row["Phi[deg]"]),
            kappa=float(row["Kappa[deg]"]),
        )

        # Add the image to the dictionary which is used for the image batch
        image_dict[row["Filename"]] = ImagePerspective(
            width=2333,
            height=1750,
            camera=camera_right,
            position=pos,
            orientation=rot,
            crs=image_crs,
        )

# Images from the left camera
with open(DATA_DIR / "eor_camera_left.csv", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        pos = np.array([float(row["Easting"]), float(row["Northing"]), float(row["Height"])])
        rot = Rotation.from_opk_degree(
            omega=float(row["Omega[deg]"]),
            phi=float(row["Phi[deg]"]),
            kappa=float(row["Kappa[deg]"]),
        )

        # Add the image to the dictionary which is used for the image batch
        image_dict[row["Filename"]] = ImagePerspective(
            width=2333,
            height=1750,
            camera=camera_left,
            position=pos,
            orientation=rot,
            crs=image_crs,
        )
# ImageBatch Class
# Initialize the image Batch by the dictionary
images = ImageBatch(image_dict, mapper=mapper_raster._georef_array)

# Map all footprints
results = images.map_footprint(points_per_edge=4)

# Example: inspect normals and per-point GSD for one footprint (printed to stderr to keep stdout as valid JSON).
for key, result in results.items():
    if result.ok is True:
        idx = np.flatnonzero(result.mask)
        if idx.size > 0:
            print(f"Example '{key}' normals (first 3): {result.normals[idx[:3]]}", file=sys.stderr)
            if result.gsd_per_point is not None:
                print(f"Example '{key}' gsd_per_point (first 3): {result.gsd_per_point[idx[:3]]}", file=sys.stderr)
        break

geo_dict = {"type": "FeatureCollection", "features": []}

features = []
for key, result in results.items():
    if result.ok is True:
        features.append(
            {
                "type": "Feature",
                "image_name": key,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [result.coordinates.tolist()],
                },
            }
        )
geo_dict["features"] = features
print(json.dumps(geo_dict, indent=4))
