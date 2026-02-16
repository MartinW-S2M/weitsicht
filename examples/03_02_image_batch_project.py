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

pyproj_network.set_network_enabled(True)  # type: ignore

# Austrian Lambert with Vertical Datum GHA (Austrian Heights)
crs_mapper = CRS("EPSG:31287+5778")
mapper_raster = MappingRaster(
    raster_path=DATA_DIR / "dtm_epsg31287plus5778.tif",
    crs=crs_mapper,
    preload_full_raster=True,
)

# Camera Model
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
image_crs = CRS("EPSG:25833").to_3d()

# so type checker are not complaining that we are assigning ImageBase children to image_dict
image_dict: Mapping[str, ImageBase] = {}
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
            width=11664,
            height=8750,
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
            width=11664,
            height=8750,
            camera=camera_left,
            position=pos,
            orientation=rot,
            crs=image_crs,
        )
# ImageBatch Class
images = ImageBatch(image_dict, mapper=mapper_raster.georef_array)

# Map Polygon
# The polygon which should be mapped
image = images["P0009912_r4_cam1.jpg"]
mapping_result = image.map_points(np.array([[5206.7, 4710.90], [5222.9, 4713.08], [5228.6, 4680.8], [5213.5, 4677.5]]))
# The mapped points
# 605026.2651450455 5313215.714125282 244.3974141243806
# 605025.9598270534 5313215.665212166 244.39450832003422
# 605026.0423648828 5313215.07452552 244.3979704470261
# 605026.334220204 5313215.098463252 244.40098045267362

# Project coordinates on images
if mapping_result.ok is False:
    raise ArithmeticError("Points could not be mapped")

result_projection = images.project(mapping_result.coordinates, crs_s=CRS("EPSG:25833").to_3d(), only_valid=True)

if result_projection is not None:
    print("Projection on image P0009078_r1_cam1")
    if result_projection["P0009078_r1_cam1.jpg"].ok is True:
        print(result_projection["P0009078_r1_cam1.jpg"].pixels)
assert result_projection is not None  # for testing
