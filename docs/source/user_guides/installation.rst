.. _installation:

Installation
============


The source code is hosted on GitHub: https://github.com/MartinW-S2M/weitsicht


Install from PyPI (recommended)
-------------------------------
If ``weitsicht`` is available on PyPI, install it via:

::

  pip install weitsicht


Install from source
-------------------
In the repository root directory, execute:
::

  pip install .

Optional extras:

::

  # tests
  pip install .[test]

  # documentation build dependencies
  pip install .[docs]

Testing uses ``pytest`` and lives in the ``tests`` folder.


Dependencies
------------

Python
^^^^^^
weitsicht runs on **Python 3.10+**.
weitsicht has been tested on Windows, Linux and MacOS, and probably also runs on other Unix-like platforms.
There are no special platform dependent binaries used in that library, but may be needed for subpackages.

Packages
^^^^^^^^
It relies only on well-developed packages. See packages for detailed description.

    * `numpy <https://www.numpy.org>`_
    * `pyproj - Python interface to proj <https://github.com/pyproj4/pyproj>`_
    * `rasterio - Easy access to geospatial raster <https://github.com/rasterio/rasterio>`_
    * `shapely <https://github.com/shapely/shapely>`_
    * `trimesh <https://github.com/mikedh/trimesh>`_

Documentation
-------------
The documentation is built with Sphinx. In the ``docs`` folder run:
::

  # Linux/macOS
  make html

  # Windows
  .\\make.bat html
