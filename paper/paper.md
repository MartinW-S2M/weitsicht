---
title: "weitsicht: Python framework and utilities for monoplotting and projecting coordinates using geo-referenced imagery."
tags:
  - Python
  - photogrammetry
  - monoplotting
  - georeference
  - geospatial
  - drones
  - wildlife monitoring
  - archaeology
authors:
  - name: Martin Wieser
    orcid: 0009-0005-9870-6494
    corresponding: true
    affiliation: 1
affiliations:
  - name: Department of Geodesy and Geoinformation, TU Wien, Austria
    index: 1
    ror: 04d836q62
date: 01 October 2026
bibliography: paper.bib
---

# Summary

**`weitsicht`** is an Apache-2.0 Python library for working with geo-referenced imagery by connecting image pixel coordinates to 3D world coordinates. It implements (i) projection of 3D coordinates into a camera model and (ii) single-image ray/surface intersection ("monoplotting") to map image measurements to 3D points using terrain representations such as planes, digital surface models (rasters), or 3D meshes. **`weitsicht`** is designed to leverage direct georeferencing (e.g., IMU/GNSS/INS metadata from drones) or camera poses computed externally (e.g., from photogrammetry/SfM pipelines) and to provide a compact, GIS-oriented API with coordinate reference system (CRS) handling via `pyproj` [@pyproj], raster handling via `rasterio` [@rasterio] (and therefore GDAL [@gdal]), and numerical foundations in `NumPy` [@numpy]. The source code for **`weitsicht`** has been archived to zenodo with the associated DOI: https://doi.org/10.5281/zenodo.20443144

# Statement of need

Aerial imagery is widely used in wildlife monitoring and ecological surveys, where researchers often require rapid,
repeatable conversion of image detections into mapped coordinates for abundance estimation and spatial modelling.
**`weitsicht`** draws on several research and software roots; one major origin is the geometric foundation developed for **WISDAMapp (Wildlife Imagery Survey — Detection and Mapping)**[@wisdamapp], a GUI used by marine and terrestrial megafauna researchers to digitize objects in images, enrich them with metadata, and map them to real-world coordinates without requiring deep photogrammetry expertise. In these operational settings, an approximate direct georeference (from INS/IMU+GNSS logs or image EXIF/XMP) is frequently sufficient, while full 3D reconstruction with Structure-from-Motion (SfM) can be unnecessarily complex, time-consuming, or prohibitively expensive for non-specialists. During a major refactoring leading up to the first public release of WISDAMapp, its geometric core functions were separated from the GUI and, together with functionality from other projects, formed **`weitsicht`** .

Beyond ecology, single-image mapping is also valuable in airborne archaeology and rapid documentation, where an INS
solution may provide direct georeferencing for each exposure but only sparse or single images are available. **`weitsicht`** supports these workflows by extracting pose and camera information from metadata, applying camera distortion models, and intersecting rays with user-provided terrain models.

For offshore surveys (a common operating environment for WISDAMapp users), SfM can fail due to low texture, specular
reflections, and moving water surfaces; in contrast, direct georeferencing combined with monoplotting can still provide
usable footprints, center points, and mapped measurements.

**`weitsicht`** also supports mapping pixels onto existing 3D surfaces, motivated in part by the **INDIGO** graffiti
documentation project [@indigo], where image-based annotations can be projected onto a 3D model for spatial analysis and dissemination.

# State of the field

A broad ecosystem of photogrammetry and SfM tools exists, including open-source packages such as COLMAP [@colmap],
OpenMVG [@openmvg], MicMac [@micmac], OpenDroneMap [@opendronemap], and commercial suites such as Agisoft Metashape and
Pix4D. These systems excel at estimating camera geometry from multi-view imagery and producing dense reconstructions, but they typically present a steep learning curve for students and domain scientists and focus on the reconstruction stage rather than lightweight downstream operations on already geo-referenced images (e.g., computing footprints, intersecting rays with arbitrary terrain models, or projecting mapped features back into images for re-sighting workflows).

At a lower level, computer-vision libraries such as OpenCV [@opencv] provide camera models, distortion, and projection
primitives. However, assembling a complete monoplotting pipeline from such building blocks (including robust CRS
transformations, metadata-driven direct georeferencing, and interchangeable terrain backends) remains substantial work
for non-expert users. **`weitsicht`** fills this gap by packaging established photogrammetric geometry into a small set of
composable abstractions aimed at geospatial workflows.

# Software design

**`weitsicht`** is organized as modular subpackages:

- `weitsicht.camera`: perspective camera models (including an OpenCV-style distortion model) used for projection and ray generation.
- `weitsicht.image`: image models combining camera intrinsics, exterior orientation, and CRS context, exposing `project_*` and `map_*` methods for 3D↔2D transformations.
- `weitsicht.mapping`: interchangeable mappers that define the terrain/surface used for ray intersections, including a horizontal plane mapper, a raster/DSM mapper based on `rasterio` [@rasterio] (and therefore GDAL [@gdal]), and a mesh mapper based on `trimesh` [@trimesh].
- `weitsicht.metadata`: utilities to estimate camera intrinsics and extract exterior orientation and CRS information from EXIF/XMP metadata (e.g., drone payloads that record IMU/GNSS/INS).
- `weitsicht.transform`: CRS and rotation utilities with a consistent "always x/y" convention to reduce common user errors when transforming between geographic and projected coordinate systems.

Modularity is an explicit design goal: new camera models (e.g., fisheye or panoramic/360° cameras) and new mapping
backends (e.g., point-cloud ray intersections or tiled/streamed raster sources such as cloud-optimized GeoTIFFs) can be
implemented by extending the corresponding camera or mapper interfaces while keeping the image-level API stable.

This design enables users to start from minimal inputs (an image, metadata, and a terrain model) and progress to mapped
3D points, footprints, or back-projected image coordinates with clear error handling and typed results. Advanced users can extend the library by implementing additional camera models, metadata backends, or mapping surfaces while reusing the same projection and CRS infrastructure.

# Research impact statement

By separating the geometric foundation from the WISDAM GUI, **`weitsicht`** provides reusable core functionality for multiple applied research domains. Within WISDAMapp [@wisdamapp], it supports workflows for wildlife monitoring (e.g., marine megafauna and dugong surveys) using drones and crewed aircraft, enabling researchers to map observations from geo-referenced imagery and to project mapped locations back into images to facilitate re-sighting and quality control. WISDAM was developed iteratively: earlier releases under the names `OceanMapper` (a GUI-less monoplotting prototype, see [@CLEGUER2021]) and `DugongDetector` (an initial, dugong-focused GUI) represent development stages of the same software that is now released as WISDAM. The workflow builds on UAV/aerial imagery survey methodology and detection-probability estimation [@HODGSON2017]. The software framework has been used in peer-reviewed research, for example in a recent dugong habitat assessment using paired drone and in-water surveys [@SAID2025] and [@Digdo2025] report a drone survey for dugongs in a limited-resource context that combined manual analysis with AI-assisted detection using WISDAM. The WISDAM app has been developed and applied in collaborations involving institutions including Edith Cowan University (Centre for Marine Ecosystems Research), Murdoch University (Centre for Sustainable Aquatic Ecosystems, Harry Butler Institute), TU Wien (photogrammetry and remote sensing), and James Cook University (TropWATER), and in projects with partners such as the Seychelles Islands Foundation, Insitu Pacific Inc., the Australian Antarctic Division and Queensland University of Technology as documented on the project website [@wisdamapp].

In archaeological and cultural-heritage contexts, **`weitsicht`** supports direct-georeferencing and single-image mapping workflows relevant to projects such as it was implemented by ARAP (University of Vienna) [@arap] and INDIGO [@indigo], where efficient projection between images and 3D surfaces enables rapid documentation, spatial analysis, and dissemination without requiring a full SfM pipeline [@WIESER2024; @WIESER2014; @DONEUS2016; @VERHOEVEN2013]. Parts of functionality developed in that projects are some of the roots beside `WISDAMapp` and where consequently enhanced and improved.

# AI usage disclosure

Generative AI was used for assistance and then edited and verified by the author.

Tool used: ChatGPT (model GPT-5.2) for documentation, code, and CI/CD maintenance.

Scope of assistance: docstring harmonization; spelling and grammar checks in documentation; drafting the latest function for advanced GSD (Ground Sampling Distance) estimation; assistance creating CI/CD workflows for GitHub Actions.

I confirm as the author that human authors reviewed, edited and validated all AI-assisted outputs and made the core design decisions.

# Acknowledgements

We acknowledge contributions and support from the Research Unit Photogrammetry of the Department of Geodesy and Geoinformation - TU Wien, specially Camillo Ressl, Norbert Pfeifer and Wilfried Karel.

We also acknowledge the support and input from Amanda Hodgson (Edith Cowan University, Perth, Western Australia, AU).


# References
