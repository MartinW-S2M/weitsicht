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

from weitsicht import CoordinateTransformer, ImageBatch, MappingHorizontalPlane, PyExifToolTags, image_from_meta

DATA_DIR = Path(__file__).parent.parent.resolve() / "examples" / "data"
pyproj_network.set_network_enabled(True)  # type: ignore


def test_import_eor_example():

    meta_data_dict = {}
    for file in (DATA_DIR / "dugong_survey").glob("*.json"):
        meta_data = json.load(open(file))
        meta_data_dict[file.stem] = meta_data

    image_dict = {}
    image_dict_utm = {}
    for key, meta_data in meta_data_dict.items():
        tags = PyExifToolTags(meta_data)
        img_res = image_from_meta(tags)  # accepts MetaTagsBase or MetaTagAll
        if img_res.ok is False:
            raise RuntimeError(f"Failed to build image '{key}' from metadata: {img_res.error} ({img_res.issues})")
        image_dict[key] = img_res.image

        img_res_utm = image_from_meta(tags, to_utm=True)  # this images are the same but the EOR will be converted to UTM
        if img_res_utm.ok is False:
            raise RuntimeError(f"Failed to build image '{key}' from metadata: {img_res_utm.error} ({img_res_utm.issues})")
        image_dict_utm[key] = img_res_utm.image

    # Mapper class
    crs_mapper = CRS("EPSG:4326+3855")
    mapper_0h = MappingHorizontalPlane(plane_altitude=0.0, crs=crs_mapper)
    # Image Batch
    # Initialize image batch class
    # During the that call all images will be assigned that mapper if no mapper is present for a single image

    images = ImageBatch(images=image_dict, mapper=mapper_0h)
    images_utm = ImageBatch(images=image_dict_utm, mapper=mapper_0h)

    # Mapping of point
    # In that example a dugong was found in one of the images and digitized by hand or by AI.
    # The center of the objects which was digitized in `image003.jpg` is:
    # image coordinates in pixel need to be a numpy array of size Nx2.
    object_digitized = np.array([[1292, 564]])

    # The ImageBatch itself has no "map_points" as it makes no sense to map for each image the same pixels.
    # images['image003'] is the single image class.
    result_mapping = images["image003"].map_points(object_digitized)

    result_mapping_utm = images_utm["image003"].map_points(object_digitized)

    if result_mapping.ok is True:
        coo = result_mapping.coordinates
        gsd = result_mapping.gsd
        crs_mapped_point = result_mapping.crs
        print("Mapped coordinate(s):", result_mapping.coordinates[result_mapping.mask])
        print("Normals (valid):", result_mapping.normals[result_mapping.mask])
        if result_mapping.gsd_per_point is not None:
            print("GSD per point (valid):", result_mapping.gsd_per_point[result_mapping.mask])
        print("Mean GSD:", gsd)
    else:
        raise ValueError("Result mapping should have worked")

    # Result if we using images with EOR in UTM will also deliver mapped points in UTM
    if result_mapping_utm.ok is True:
        coo_utm = result_mapping_utm.coordinates
        gsd = result_mapping_utm.gsd
        crs_mapped_point_utm = result_mapping_utm.crs
        print("Mapped coordinate(s):", result_mapping_utm.coordinates[result_mapping_utm.mask])
        print("Normals (valid):", result_mapping_utm.normals[result_mapping_utm.mask])
        if result_mapping_utm.gsd_per_point is not None:
            print("GSD per point (valid):", result_mapping_utm.gsd_per_point[result_mapping_utm.mask])
        print("Mean GSD:", gsd)
    else:
        raise ValueError("Result mapping should have worked")

    ct = CoordinateTransformer.from_crs(result_mapping.crs, result_mapping_utm.crs)

    assert ct is not None
    pt_ecef_to_utm = ct.transform(result_mapping.coordinates)
    np.testing.assert_allclose(pt_ecef_to_utm, result_mapping_utm.coordinates, atol=0.005)
    # For both batches we will project points in both CRS
    # Project once the ECEF
    # Second project UTM
    # They should be very similar in the range of lower that 0.1 pixel difference
    result_projections = images.project(coo, crs_mapped_point)

    res_1 = []
    if result_projections is not None:
        print("All projections")
        for key, prj_result in result_projections.items():
            if prj_result.ok is True:
                print(key, prj_result.pixels)
                res_1.append(prj_result.pixels)
    assert result_projections is not None  # for testing

    result_projections = images.project(coo_utm, crs_mapped_point_utm)

    res_2 = []
    if result_projections is not None:
        print("All projections")
        for key, prj_result in result_projections.items():
            if prj_result.ok is True:
                print(key, prj_result.pixels)
                res_2.append(prj_result.pixels)
    assert result_projections is not None  # for testing

    # Now use UTM images
    result_projections = images_utm.project(coo, crs_mapped_point)

    res_3 = []
    if result_projections is not None:
        print("All projections")
        for key, prj_result in result_projections.items():
            if prj_result.ok is True:
                print(key, prj_result.pixels)
                res_3.append(prj_result.pixels)
    assert result_projections is not None  # for testing

    result_projections = images_utm.project(coo_utm, crs_mapped_point_utm)

    res_4 = []
    if result_projections is not None:
        print("All projections")
        for key, prj_result in result_projections.items():
            if prj_result.ok is True:
                print(key, prj_result.pixels)
                res_4.append(prj_result.pixels)
    assert result_projections is not None  # for testing

    tol_px = 0.1

    a1 = np.stack(res_1)  # (n_images, n_points, 2)
    a2 = np.stack(res_2)
    a3 = np.stack(res_3)
    a4 = np.stack(res_4)

    assert a1.shape == a2.shape == a3.shape == a4.shape
    np.testing.assert_allclose(a2, a1, atol=tol_px, rtol=0)
    np.testing.assert_allclose(a3, a1, atol=tol_px, rtol=0)
    np.testing.assert_allclose(a4, a1, atol=tol_px, rtol=0)
