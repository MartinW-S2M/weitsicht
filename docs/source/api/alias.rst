.. currentmodule:: weitsicht

------------
Types Alias
------------
Type Alias used in the API. This are numpy aliases to state the dimension and type of ``np.ndarray``
The definition is coded in utils.py

:type:`Vector2D` numpy array of shape (2,) and dtpye `float`; eg. numpy.array([100.5, 20.0])

:type:`Vector3D` numpy array of shape (3,) and dtpye `float`; eg. numpy.array([100.5, 20.0, 50.1])

:type:`MaskN_` numpy array of shape (N,) and dtype `bool`; eg. numpy.array([True, True, False, ..., True])

:type:`ArrayN_` numpy array of shape (N,) and dtype `float`; eg. numpy.array([100.0, 20.0, 50.0, ..., 45.3])

:type:`Array3x3` numpy array of shape (3,3) and dtype `float`; eg. numpy.eye(3,3,dtype=float)

:type:`ArrayNx2` numpy array of shape(N,2) and dtype `float` -> N rows and 2 columns

:type:`ArrayNx3` numpy array of shape(N,3) and dtype `float` -> N rows and 3 columns
