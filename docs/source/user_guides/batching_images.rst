=================
Batching Images
=================

``ImageBatch`` groups multiple geo-referenced images, optionally sharing a common mapper. It provides convenience methods to project, map footprints, and centers in bulk while preserving per-image validity masks.
This is a container class which can hold multiple images. :ref:_api_image_batch_

For example if you have a lots of images of a drone flight you can add all images to the batch
and find all images which contain the projection of a 3D coordinate.


When to use
-----------
- You have many images that should use the same mapper.
- You want a single call to project or map footprints and get results per image.
- You need masking to quickly filter which images see a given 3D point set.

Creating a batch
----------------
.. code-block:: python

    from weitsicht import ImageBatch

    batch = ImageBatch({'img1': image1, 'img2': image2}, mapper=shared_mapper)
    # images inherit the mapper if they don't have one

Core methods
------------
- ``__len__``: number of images.
- ``add_images(images: dict[str, ImageBase])``: extend the batch; raises on duplicate keys.
- ``__getitem__(keys)`` and ``index(indices)``: retrieve single or multiple images by name or index order.
- ``map_center_point(mapper=None)`` -> ``dict[name, MappingResult]``: principal point/center per image (see masks and issues below).
- ``map_footprint(points_per_edge=0, mapper=None)`` -> ``dict[name, MappingResult]``: footprint polygons at optional edge densification.
- ``project(coordinates, crs_s=None, only_valid=False)`` -> ``dict[name, ProjectionResult] | None``: project 3D points into each image; with ``only_valid`` drops images where every point is out of frame.

Results & issues
----------------
- Each value in the returned dict is a ``MappingResult`` or ``ProjectionResult`` (see ``results_prj_mapping.rst`` for full fields).
- Always check ``result.ok``; then apply ``result.mask`` to filter valid rows.
- ``issues`` is a set of ``Issue`` enums that can appear even when ``ok`` is ``True`` (partial success). Typical cases in batching:
  - ``OUTSIDE_RASTER`` / ``WRONG_DIRECTION`` / ``RASTER_NO_DATA`` / ``MAX_ITTERATION`` from raster mappers, and ``NO_INTERSECTION`` from mesh mappers.
  - for image batch there is special issue ``IMAGE_BATCH_ERROR`` indicating single image function raised error. (see ``results_prj_mapping.rst`` section Image Batch).
- When ``only_valid=True`` in ``project``, all image where at least a single projection is found. If none of the images contain the projection it will return ``None``.

Usage patterns
--------------
- Share a mapper once: pass ``mapper`` to ``ImageBatch``; missing per-image mappers are filled in.
- Filter quickly: call ``project(..., only_valid=True)`` to keep only images that see at least one point.


Initialize class
----------------
The ImageBatch class is initialized with a dictionary containing single images. The dict key can be any string,
for example the image name or the image path.

.. code-block:: python

    # Create the ImageBatch class which contains the images

    image_class1 = ImagePerspective(..., mapper=HorizontalMapper(...))
    image_class2 = ImagePerspective(...)  # could as well be an orthophoto
    # The dictionary keys are strings, for example image name or image path
    image_dict = {"DSC001.jpg": image_class1, "DSC002.jpg": image_class2}

    # The mapper is not mandatory. the mapper specified here overwrites the mappers specified within images of the batch

    images = ImageBatch(image_dict, mapper=mapper_raster)


.. hint::
    If a mapper is specified at initialization or assigned later to the image batch it will be used
    for mapping methods for every image and assigns that mapper for single images.

    If no mapper is stated for the batch or methods, the single image's mapper will be used for mapping methods.


Mapping Center Point and Footprint
---------------------------------------
Will return a dictionary with keys of self.images and the result of the single image's corresponding method.

.. code-block:: python


    # Mapper can be stated optional. Will be used instead of mapper specified for batch and single images
    images.map_center_point(mapper=...)

    # points_per_edge: The number of points which should be inserted between corners. 0:only corner points are mapped
    images.map_footprint(points_per_edge=3, mapper=...)


Project
-------------
Project coordinates into the image.

Will return a dictionary with keys of self.images and the result of the single image's project method.

With only_valid=True the returned dict will only contain images where for at least one coordinate the projection is within the image extend.
.. code-block:: python

    # coordinates are the coordinates which should be projected
    # crs_s is the coordinate reference system of the coordinates.
    images.project(coordinates: ArrayNx3, crs_s: CRS | None = None, only_valid: bool = False)



Adding images
-------------
Duplicate keys are rejected (``KeyError``) and nothing is added if a clash occurs.

.. code-block:: python

    images_to_add={}
    images_to_add['img1']=IMAGEPerspective(width=2333, height=1750,
                                                           camera=camera_left,
                                                           position=pos_img1, orientation=rot_img1,
                                                           crs=image_crs)

    images_to_add['img2']=IMAGEPerspective(width=2333, height=1750,
                                                           camera=camera_left,
                                                           position=pos_img2, orientation=rot_img2,
                                                           crs=image_crs)

    batch.add_images(images_to_add)

    # Trying to add the same keys will raise a key error
    try:
        batch.add_images(images_to_add)  # same keys -> KeyError
    except KeyError as e:
        print(e)


Access single images
--------------------
Use dictionary access to reach per-image attributes or methods.

.. code-block:: python

    img = images['img1']
    crs = img.crs
    px, mask = img.project(np.array([[604566.25, 5313155.36, 200]]), crs_s=CRS('EPSG:25833+5778'))


Indices and keys
---------------------------
From the image batch, images can be extracted as dictionary [key, Image]:

- Getter via ``[]``: use a single key or a list/tuple of keys from the dictionary used to create the batch.
- Getter via ``.index()``: use a single index or list/tuple of indices.

.. code-block:: python

    # Get single image (ImageClass )
    img = images['P0009065_r1_cam1.jpg']

    # Get a dictionary of images
    img_subset = images['P0009065_r1_cam1.jpg', 'P0009066_r1_cam1.jpg']
    #will return {'P0009065_r1_cam1.jpg': <..IMAGEPerspective..>, 'P0009066_r1_cam1.jpg': <..IMAGEPerspective..>}
    # or use
    # images[['P0009065_r1_cam1.jpg', 'P0009066_r1_cam1.jpg']]
    # images[('P0009065_r1_cam1.jpg', 'P0009066_r1_cam1.jpg')]

    # Get a dictionary of images
    img = images.index(0)
    img = images.index((0,1))
    #will return {'P0009065_r1_cam1.jpg': <..IMAGEPerspective..>, 'P0009066_r1_cam1.jpg': <..IMAGEPerspective..>}
    # or use
    # images.index([0,1])


Modify an image
---------------
Assign a new geo-referenced image to an existing key (must subclass ``ImageBase``).

.. code-block:: python

    images['img1'] = image_updated

.. note::
   When replacing images like this, no mapper is injected automatically. Set ``image_updated.mapper`` (or pass ``mapper=`` when you build it) if it should share the batch mapper.

Example end-to-end
------------------
.. code-block:: python

    # given: images dict already geo-referenced, mapper already loaded
    batch = ImageBatch(images, mapper=mapper)

    # Footprints
    footprints = batch.map_footprint(points_per_edge=2)

    # Projection of 3D targets
    pts_world = np.array([[52005, 65200, 133.05], [52010, 65260, 123.55]])
    projections = batch.project(pts_world, crs_s=crs_coordinates, only_valid=True)

    # Centers
    centers = batch.map_center_point()

Notes
-----
- Images must already be geo-referenced (camera, pose, CRS set) for mapping/projection to work.
- If per-image mapper differs, assign mapper to image before batching or pass ``mapper=None`` and let each image keep its own.
- Returned masks are per-point booleans; use them to drop out-of-frame results.
