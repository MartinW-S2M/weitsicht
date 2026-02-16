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

"""Utilities for intersecting lines with a regular grid."""

import numpy as np

__all__ = ["vector_projection", "raster_index_p1_p2", "line_grid_intersection_points"]


def vector_projection(p1, p2, g_1):
    """Project points onto a line defined by two points.

    :param p1: First point defining the line (x, y).
    :type p1: array-like
    :param p2: Second point defining the line (x, y).
    :type p2: array-like
    :param g_1: Points to project, shape ``(N, 2)``.
    :type g_1: array-like
    :return: Projected points, shape ``(N, 2)``.
    :rtype: numpy.ndarray
    """
    b = np.zeros(g_1.shape, dtype=float) + (p2 - p1)
    a = g_1 - p1

    return (np.einsum("ij,ij->i", a, b) / np.einsum("ij,ij->i", b, b) * b.T).T + p1


def line_grid_intersection_points_iterative(p_line_1, p_line_2):  # pragma: no cover
    """This works but need more testing and edge case checking"""
    x1 = p_line_1[0]
    y1 = p_line_1[1]
    x2 = p_line_2[0]
    y2 = p_line_2[1]

    _x = np.floor(x1) + 0.5
    _y = np.floor(y1) + 0.5

    _x_end = np.floor(x2) + 0.5
    _y_end = np.floor(y2) + 0.5

    if np.isclose(abs(x2 - x1), 0):
        _x_dir = 1
    else:
        _x_dir = np.round((x2 - x1) / abs(x2 - x1))

    if np.isclose(abs(y2 - y1), 0):
        _y_dir = 1
    else:
        _y_dir = np.round((y2 - y1) / abs(y2 - y1))

    u_vec = (p_line_2 - p_line_1) / np.linalg.norm(p_line_1 - p_line_2)

    list_index = [[_x, _y]]
    while True:
        c1 = np.array([_x + _x_dir, _y])
        c2 = np.array([_x, _y + _y_dir])

        _d_c1 = np.linalg.norm(np.cross((c1 - p_line_1), u_vec))
        _d_c2 = np.linalg.norm(np.cross((c2 - p_line_1), u_vec))
        if _d_c1 < _d_c2:
            _x = _x + _x_dir

        else:
            _y = _y + _y_dir

        list_index.append([_x, _y])

        if (_x == _x_end) and (_y == _y_end):
            break

    return np.floor(np.array(list_index)).astype(int)


def raster_index_p1_p2(p_line_1, p_line_2):
    """Return the bounding raster indices covering a line segment.

    :param p_line_1: Start point (x, y).
    :type p_line_1: array-like
    :param p_line_2: End point (x, y).
    :type p_line_2: array-like
    :return: Tuple ``(low_x, high_x, low_y, high_y)`` as integer indices.
    :rtype: tuple[int, int, int, int]
    """
    x1 = p_line_1[0]
    y1 = p_line_1[1]
    x2 = p_line_2[0]
    y2 = p_line_2[1]

    if x1 <= x2:
        low_x = int(np.floor(x1))
        high_x = int(np.ceil(x2))
    else:
        high_x = int(np.ceil(x1))
        low_x = int(np.floor(x2))

    if y1 <= y2:
        low_y = int(np.floor(y1))
        high_y = int(np.ceil(y2))

    else:
        high_y = int(np.ceil(y1))
        low_y = int(np.floor(y2))

    return low_x, high_x, low_y, high_y


def line_grid_intersection_points(p_line_1, p_line_2):
    """Compute raster cell indices intersected by a line segment.

    :param p_line_1: Start point (x, y).
    :type p_line_1: array-like
    :param p_line_2: End point (x, y).
    :type p_line_2: array-like
    :return: Array of integer indices with shape ``(N, 2)``.
    :rtype: numpy.ndarray
    """
    # Suffers a little from floating point
    # Would need to check epsilon before doing floor
    # Should then include both cells just for safety

    x1 = p_line_1[0]
    y1 = p_line_1[1]
    x2 = p_line_2[0]
    y2 = p_line_2[1]

    if x1 <= x2:
        step_x = 1
        start_x = np.floor(x1)
        stop_x = np.ceil(x2) + 1
    else:
        step_x = -1
        start_x = np.ceil(x1)
        stop_x = np.floor(x2) - 1

    if y1 <= y2:
        step_y = 1
        start_y = np.floor(y1)
        stop_y = np.ceil(y2) + 1

    else:
        step_y = -1
        start_y = np.ceil(y1)
        stop_y = np.floor(y2) - 1

    # 1st vertical rows

    x3_matrix = np.arange(start_x, stop_x, step_x)
    x4_matrix = x3_matrix * 1.0
    # _size = np.ceil(x2) - np.floor(y1) + 1
    y3_matrix = np.ones(x3_matrix.shape) * start_y
    y4_matrix = np.ones(x3_matrix.shape) * stop_y

    p_x = np.array([]).reshape(0, 2)
    _d = (x1 - x2) * (y3_matrix - y4_matrix) - (y1 - y2) * (x3_matrix - x4_matrix)
    if not np.any(_d == 0.0):
        _t_s = (x1 - x3_matrix) * (y3_matrix - y4_matrix) - (y1 - y3_matrix) * (x3_matrix - x4_matrix)

        _u_s = (x1 - x2) * (y1 - y3_matrix) - (y1 - y2) * (x1 - x3_matrix)

        t = _t_s / _d
        _u = -_u_s / _d

        p_x = np.array([x3_matrix, y1 + t * (y2 - y1)]).T

        # If it's going to the left we need to switch the x coordinates otherwise we will miss cells
        if y1 > y2:
            p_x[:, 0] -= 0.1

    # 2nd vertical rows

    y3_matrix = np.arange(start_y, stop_y, step_y)
    y4_matrix = y3_matrix * 1.0
    x3_matrix = np.ones(y3_matrix.shape) * start_x
    x4_matrix = np.ones(y3_matrix.shape) * stop_x

    p_y = np.array([]).reshape(0, 2)
    _d = (x1 - x2) * (y3_matrix - y4_matrix) - (y1 - y2) * (x3_matrix - x4_matrix)
    if not np.any(_d == 0.0):
        _t_s = (x1 - x3_matrix) * (y3_matrix - y4_matrix) - (y1 - y3_matrix) * (x3_matrix - x4_matrix)
        _u_s = (x1 - x2) * (y1 - y3_matrix) - (y1 - y2) * (x1 - x3_matrix)

        t_vertical = _t_s / _d
        _u_vertical = -_u_s / _d

        p_y = np.array([x1 + t_vertical * (x2 - x1), y3_matrix]).T

        if x1 > x2:
            p_y[:, 1] -= 0.1

    p_all = np.vstack((p_x, p_y))

    # start and endpoint are within one cell
    if len(p_all) == 0:
        p_all = np.array([[x1, y1]])
        return np.floor(p_all).astype(int)

    _dist = np.linalg.norm(p_all - np.array([x1, y1]), axis=1)
    idx_sort_deist = np.argsort(_dist)

    p_all = p_all[idx_sort_deist, :]

    return np.floor(p_all).astype(int)
