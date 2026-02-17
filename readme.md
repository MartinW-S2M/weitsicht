<p align="center">
  <a href=https://github.com/MartinW-S2M/weitsicht>
  <img src="https://github.com/MartinW-S2M/weitsicht/blob/main/logos/weitsicht.svg?raw=true" alt="weitsicht logo" width="240">
</p>

# weitsicht
[![pypi](https://img.shields.io/pypi/v/weitsicht)](https://pypi.org/project/weitsicht/)
![Python Version 3.10,3.11,3.12,3.13](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?style=flat&logo=python&logoColor=white)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)


[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![pep8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)
![PyPI - Types](https://img.shields.io/pypi/types/weitsicht)

![tests](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MartinW-S2M/43e035fba612a7cf89ff180d3a41fc2f/raw/test_count.json)
![cov](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MartinW-S2M/43e035fba612a7cf89ff180d3a41fc2f/raw/covbadge.json)
&emsp;&emsp;![loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MartinW-S2M/43e035fba612a7cf89ff180d3a41fc2f/raw/loc.json)
![comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MartinW-S2M/43e035fba612a7cf89ff180d3a41fc2f/raw/comments.json)

[![Tests](https://github.com/MartinW-S2M/weitsicht/actions/workflows/tests.yml/badge.svg)](https://github.com/MartinW-S2M/weitsicht/actions/workflows/tests.yml)
&emsp;&emsp;
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/MartinW-S2M/weitsicht/main.svg)](https://results.pre-commit.ci/latest/github/MartinW-S2M/weitsicht/main)
&emsp;&emsp;![Docs passing](https://app.readthedocs.org/projects/weitsicht/badge/?version=latest)


**Python package to use the direct geo-reference information of images for mapping and projection.**

Its designed to simplify the use and implementation of functions and classes needed all the way from image points to mapped
3d points or the other way around from 3d point to image points.
Additionally, it is easy to get information like mapped image footprints, center points or transform them to other coordinate systems. Single-image ray intersection with a ground/3D model is often called **Monoplotting** (in ``weitsicht`` this is what the image ``map_*`` methods like ``map_points`` / ``map_center_point`` / ``map_footprint`` do with a mapper).

There are classes for camera models, images, mapping functions and additional utils which simplify and abstract to a level where non-photogrammetry experts can work with it.

Currently, it is possible to use perspective and ortho-imagery and for mapping a horizontal plane, rasters (e.g. DSM) and meshes.

<table>
  <tr>
    <td><img src="https://github.com/MartinW-S2M/weitsicht/blob/main/docs/source/_static/raster_mapping.jpg?raw=true" alt="Raster mapping example" width="80%"></td>
    <td><img src="https://github.com/MartinW-S2M/weitsicht/blob/main/docs/source/_static/mesh_pic.jpg?raw=true" alt="Mesh example" width="80%"></td>
    <td><img src="https://github.com/MartinW-S2M/weitsicht/blob/main/docs/source/_static/example_images/image_batch_footprints.jpg?raw=true" alt="Image batch footprints" width="100%"></td>
  </tr>
</table>

## Why is it called weitsicht?

`weitsicht` is a German word that roughly means "far-sight" - being able to see into the distance. That fits the core idea here: a photo isn't just pixels; it's an image plane anchored to a viewpoint, and we use geometry to connect that plane to the world beyond the camera. In other words, it's applied photogrammetry with a bit of home-brew minimalism: point, project, monomplot. But "weitsicht" also means having foresight - building with tomorrow's applications and datasets in mind, not just today's demo. So the library stays modular: camera models, mappers, and metadata backends are plug-in pieces you can remix instead of rewriting. Call it far-sight for imagery, and long-sight for architecture.

## Capabilities

- **Monoplotting/Mapping**, map the image’s center-point and footprint (image extend) or image point easily.
- **Projection**, get the pixel position of 3D coordinates.
- **CRS**, weitsicht handles coordinate system conversions (to some extend)
- **Perspective Image and Camera**, mathematic model of your digital camera and pose.
- **Ortho imagery**, use ortho imagery to map contant or convert 2D coordinates to 3D.
- **Mapper Classes**, several mapper classes can be used to map your pixel data: HorizontalPlane, Raster, Mesh
- **ImageBatch**, container class to perform tasks on multiple images. Find all images where coordinates are visible. Map for all images footprint and centerpoint.
- **Meta-Data**, use image’s meta-data (EXIF, XMP) to estimate camera model and image pose.


## Table of Contents
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Documentation](#documentation)
- [Brief History](#brief-history)
- [Goals](#goals)
- [Future Plans](#future-plans)
- [Contribution](#contribution)
- [Discussion](#discussion)
- [Package Structure](#package-structure)
- [Example Usage](#example-usage)
- [Notes on PyProj](#notes-on-pyproj)
- [License](#licence)

## Installation
The source code is currently hosted on GitHub at: https://github.com/MartinW-S2M/weitsicht

### PyPI
Binary installers for the latest released version are available at the [Python
Package Index (PyPI)] https://pypi.org/project/weitsicht/
```
  pip install weitsicht
```

### From Source
In the `weitsicht` directory (same one where you found this file after cloning the git repo), execute:
```
  pip install .

  # include test dependencies
  pip install .[test]
```

Tests use ``pytest`` and live in the **tests** folder. As well all scripts within **examples** are tested.
For testing Internet access has to be available as pyproj will download needed rasters (e.g EGM2008 earth gravity model)
Alternative its possible to tell pyproj to use local stored grids/data - see [pyproj-datadir](https://pyproj4.github.io/pyproj/stable/api/datadir.html)

### Installation of 3rd-party dependencies
To provide a full simple workflow a package to read metadata is needed. see [Dependencies](#dependencies)

## Dependencies

### Python
weitsicht runs on **Python 3.10+**.
weitsicht has been tested on Windows, Linux and MacOS, and probably also runs on other Unix-like platforms.

### Packages
  It relies only on well-developed packages
- [numpy](https://www.numpy.org)
- [pyproj - Python interface to proj](https://pyproj4.github.io/pyproj/stable)
- [rasterio - Easy access to geospatial raster](https://rasterio.readthedocs.io/en/stable)
- [shapely](https://shapely.readthedocs.io/en/stable/index.html)
- [trimesh](https://trimesh.org/)

Additionally, to provide a workflow for drone imagery the metadata from images has to be extracted.
You can use Phil Harvey's exiftool together with [PyExifTool](https://sylikc.github.io/pyexiftool/index.html).
Although long‑term maintenance of those packages is not clear, Phil Harvey's tool provides to date the most complete metadata
reader for most image formats, including raw camera files. In ``weitsicht`` provides an interface for different exiftool wrappers, but currently only a parser for tags from PyExifTool is implemented. A exiv2 parser would be very welcome as contributions.

## Documentation
The current documentation is available at [weitsicht.readthedocs.io](https://weitsicht.readthedocs.io)
There you will also find information about the mathematical background and definitions.
ANy contributions are welcome for extending documentation.

## Brief History
weitsicht was formerly part of **WISDAM (Wildlife Imagery Survey – Detection and Mapping)** (and its predecessor DugongDetector) but is now provided as its own python package.
It is used as the photogrammetry/mapping core of WISDAMapp to map images, objects and project them back into images.
This allows the users to assign groups, spatial metadata and find resights.

**WISDAMapp (Wildlife Imagery Survey – Detection and Mapping)** is a python GUI framework based on QT for the digitization and metadata enrichment of objects
digitized in images and ortho-photos. Geo-referenced imagery can be used to map digitized objects to 3D or project them back into images for the purpose of grouping

**The WISDAMapp repository can be found under http://www.github.com/WISDAMapp/**

Currently WISDAMapp undergoes refactoring to switch from the old WISDAMcore to ``weitsicht``


## Goals
The intention of weitsicht is to provide the community an easy-to-use package to deal with geo-referenced imagery.
No matter if environmental science, geomatic, GIS or drone community more and more imagery is available (mostly from drones/UAS) which can be used to perform mapping operations.

Many users can’t leverage their imagery beyond basic tasks without photogrammetry/SfM software; we aim to bridge that gap.

## Future Plans
As the package is currently very basic, already a few topics are around which could be implemented:
+ Extend image and camera model classes (For example, 360degree imagery)
+ Extend Rotation class to other common notations.
+ Import/Conversion of the results from photogrammetry / sfm packages.
+ Improve mapping on mesh. (holes in the mesh are currently a problem)
+ Provide ability to save derived geometries.
+ Maybe switch coordinate class to use geopandas.
+ Provide ability to use network assets for mapping (e.g. Cloud optimized geotiff)


## Discussion
Use the discussion page for questions, support, and general discussion.

## Contribution
Contributions are highly welcome to extend the package.
All levels of contributions are welcome from extending the docs, examples, mathematical discussions, coding, test implementations, providing test samples.
Please find more info in CONTRIBUTION.md

## Package Structure

> [!IMPORTANT]
> weitsicht uses pyproj and using the setting always_xy for all Transformers.

weitsicht was designed with flexibility and extensibility in mind.
The library consists of a few classes, each with increasingly more features.

* ``weitsicht.camera`` is the sub package for camera models (e.g. OpenCV camera model)
It contains only the mathematical model used to transform 2D image coordinates to 3D coordinates in the camera system
and backwards. This class mostly does not need to be called itself on a basic level but is used by the image class.
New classes can be easily implemented or extended.


* ``weitsicht.image`` is the subpackage which provides image classes. It is used to deal with the geo-reference
information which states the image pose in 3D space. Main functions of the classes are:
  * project: Project 3D coordinates into image 2D space
  * map_points: Map pixel points into 3D space using a provided mapper.


* ``weitsicht.mapping`` is the subpackage which provides mapping classes.
Currently, mapping bases on a horizontal plane (mappingPlane) and mapping based on raster (mappingRaster) using rasterio.
Main functions of the classes are:
  * map_heights_from_coordinates: Get the height of coordinates.
  * map_coordinates_from_rays: Get the intersection coordinates of a ray in 3D space


* ``weitsicht.transform`` is the subpackage which provides transformation classes and functions
  * cooTransformer class: Transform points using pyproj. Also provides an option to create geojson dict
  * utm_converter: Convert a coordinate to WGS84/utm and get crs class as well
  * rotation class: Class dealing with Rotation matrices used in photogrammetry

> [!NOTE]
> Some additional helper functions are needed to be used by WISDAM (e.g. the function to load classes by a dictionary)

## Example Usage

### Map Pixels to ground with height 0.0

    import numpy as np
    import pyproj
    from pyproj import CRS

    from weitsicht import (CameraOpenCVPerspective, ImagePerspective, MappingHorizontalPlane)
    from weitsicht import Rotation

    # To directly download the grids needed for coordinate transformation we enable the network capability of proj
    pyproj.network.set_network_enabled(True)

    # Construct Mapper
    # We use the horizontal plane mapper in the CRS System WGS84 with EGM2008 heights
    crs_mapper = CRS("EPSG:4326+3855")
    mapper = MappingHorizontalPlane(plane_altitude=0.0, crs=crs_mapper)

    # Camera Model
    # The camera model's width and height is the image shape which was used during calibration,
    # allowing the image class to use resampled images.
    cam = CameraOpenCVPerspective(width=6000, height=4000, fx=2360, fy=2360)

    # Image Pose
    position = np.array([602013.0, 5340384.696, 100.0])
    orientation = Rotation.from_opk_degree(omega=1.5, phi=5.6, kappa=65.0)

    # Image Coordinate Reference System
    crs = CRS("EPSG:25833+3855")  # UTM Zone 33
    image = ImagePerspective(width=6000, height=4000, mapper=mapper, camera=cam,
                             crs=crs, position=position, orientation=orientation)

    # Map to image coordinates
    result = image.map_points(np.array([[2000,300],[2300, 400]]))
    if result.ok:
      print("GSD of points: %f" % result.gsd)
      print("Coordinates mapped", *result.coordinates)


Refer to documentation for more examples http://weitsicht.github.io/

## Notes on PyProj
This packages heavily depends on pyproj CRS and Transformer/TransformerGroups.

> [!IMPORTANT]
> weitsicht uses the setting always_xy for all Transformers.
> Except you are constructing your own transformation pipeline

The user must take care that all data is available for the Transformation needed
([Pyproj - Datadir](https://pyproj4.github.io/pyproj/stable/api/datadir.html))  or the network option is enabled
([Pyproj - Network settings ](https://pyproj4.github.io/pyproj/stable/api/network.html#proj-network-settings)).

    pyproj.network.set_network_enabled(True)

All transformation are done by `Transformer.from_crs()`.
Standard option is to allow only best transformations and an exception will be raised otherwise (Including ballpark transformations,
[PyProj - Transformer](https://pyproj4.github.io/pyproj/stable/api/transformer.html#pyproj.transformer.Transformer.from_crs)).
To allow ballpark and allow also other than the best transformations, set:

    weitsicht.allow_ballpark_transformations() # sets cfg._ballpark_transformation to True
    # is the same as allow_ballpark_transformations(True), to disallow set to False

    weitsicht.allow_non_best_transformations() # sets cfg._only_best_transformation to False
    # is the same as allow_non_best_transformations(True), to disallow set to False

## Licence
weitsicht is licensed under the Apache License, Version 2.0 (Apache-2.0).

See `LICENSE` for the full license text.

[Go to Top](#table-of-contents)
