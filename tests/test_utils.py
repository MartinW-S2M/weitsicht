import numpy as np
import pytest

from weitsicht.utils import to_array_nx2, to_array_nx3


def test_to_array_nx2():

    with pytest.raises(ValueError):
        _ = to_array_nx2([100])

    with pytest.raises(ValueError):
        _ = to_array_nx2([[100], [100], [100]])

    np.testing.assert_allclose(to_array_nx2([100, 100, 100, 100]), np.array([[100, 100]]))


def test_to_array_nx3():

    with pytest.raises(ValueError):
        _ = to_array_nx3([100])

    with pytest.raises(ValueError):
        _ = to_array_nx3([[100], [100], [100]])

    np.testing.assert_allclose(to_array_nx2([100, 100, 100, 100]), np.array([[100, 100]]))

    with pytest.raises(ValueError):
        _ = to_array_nx3([100, 100])

    with pytest.raises(ValueError):
        _ = to_array_nx3([[100, 100], [100, 100], [100, 100]])

    arr = to_array_nx3([[100, 100], [100, 100], [100, 100]], fill_z=0)
    np.testing.assert_allclose(arr, np.array([[100, 100, 0], [100, 100, 0], [100, 100, 0]]))
