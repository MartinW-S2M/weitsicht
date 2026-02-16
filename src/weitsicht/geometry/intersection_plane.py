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

"""Ray/line intersections with planes."""

import numpy as np

from weitsicht.utils import ArrayNx3, MaskN_, Vector3D

__all__ = ["intersection_plane", "intersection_plane_mat_operation"]


def intersection_plane(
    line_vec: np.ndarray,
    line_point: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray | None = None,
):
    """Intersect a line with a plane.

    If ``plane_normal`` is ``None``, the plane normal defaults to ``(0, 0, 1)``.

    :param line_vec: Direction vector of the line.
    :type line_vec: numpy.ndarray
    :param line_point: A point on the line.
    :type line_point: numpy.ndarray
    :param plane_point: A point on the plane.
    :type plane_point: numpy.ndarray
    :param plane_normal: Plane normal vector, defaults to ``None`` (interpreted as ``(0, 0, 1)``).
    :type plane_normal: numpy.ndarray | None
    :return: Tuple ``(intersection_point, direction_valid)``. If no valid intersection exists,
        the point contains NaNs and ``direction_valid`` is ``False``.
    :rtype: tuple[numpy.ndarray, bool]
    """
    epsilon = 1e-6
    # Define plane
    if plane_normal is None:
        plane_normal = np.array([0, 0, 1])

    n_dotu = plane_normal.dot(line_vec)

    # Test for parallelity or vector insi
    if abs(n_dotu) < epsilon:
        direction_valid = False
        return np.array([np.nan, np.nan, np.nan]), direction_valid
        # print "no intersection or line is within plane"

    w = line_point - plane_point
    si = -plane_normal.dot(w) / n_dotu
    p_si = w + si * line_vec + plane_point

    v1 = line_point - p_si
    v2 = line_point - (line_point + line_vec)
    cos_a = v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2)

    if cos_a < 0.9:
        direction_valid = False
    else:
        direction_valid = True

    return p_si, direction_valid


def intersection_plane_mat_operation(
    line_vec: ArrayNx3,
    line_point: ArrayNx3,
    plane_point: Vector3D,
    plane_normal: Vector3D | None = None,
) -> tuple[ArrayNx3, MaskN_]:
    """Intersect multiple rays with a plane (vectorized).

    For every ray defined by ``line_point[i] + t * line_vec[i]`` this function computes the
    intersection with a plane defined by ``plane_point`` and ``plane_normal``.

    If ``plane_normal`` is ``None``, the plane normal defaults to ``(0, 0, 1)``.

    :param line_vec: Ray direction vectors.
    :type line_vec: ArrayNx3
    :param line_point: Ray origin points.
    :type line_point: ArrayNx3
    :param plane_point: A point on the plane.
    :type plane_point: Vector3D
    :param plane_normal: Plane normal vector, defaults to ``None`` (interpreted as ``(0, 0, 1)``).
    :type plane_normal: Vector3D | None
    :return: Tuple ``(intersection_points, valid_mask)``. Invalid intersections contain NaNs.
    :rtype: tuple[ArrayNx3, ``MaskN_``]
    """
    valid: np.ndarray
    valid = np.zeros((line_vec.shape[0],), dtype=bool)

    epsilon = 1e-7
    # Define plane
    if plane_normal is None:
        pln = np.array([0.0, 0.0, 1.0])
    else:
        pln = plane_normal
    # Define ray
    # rayDirection = numpy.array([0, -1, -1])
    # rayPoint = numpy.array([0, 0, 10])  # Any point along the ray

    n_dotu = line_vec.dot(pln)
    # np.einsum('ij,ij->i', plane_normal, line_vec)

    # check for parallel lines to plane
    valid = abs(n_dotu) > epsilon

    w = line_point - plane_point

    si = np.zeros(n_dotu.shape[0], dtype=float)
    si[valid] = np.dot(-w[valid, :], pln) / n_dotu[valid]

    # si * line_vec
    si_m_line_vec = np.zeros(w.shape, dtype=float)
    si_m_line_vec[valid, :] = np.einsum("ij,i->ij", line_vec[valid, :], si[valid])

    # p_si = np.zeros((w.shape[0], 3), dtype=float)
    p_si = np.empty((w.shape[0], 3), dtype=float)
    p_si.fill(np.nan)
    p_si[valid, :] = w[valid, :] + si_m_line_vec[valid, :] + plane_point

    # Check if the direction would be to the other direction aka backwards
    # because that is mathematically possible on a line plane intersection
    v1 = np.zeros(w.shape, dtype=float)
    v1[valid, :] = p_si[valid, :] - line_point[valid, :]
    valid[valid] = np.logical_and(valid[valid], np.linalg.norm(v1[valid, :], axis=1) > epsilon)
    cos_a = (
        np.einsum("ij,ij->i", v1[valid, :], line_vec[valid, :])
        / np.linalg.norm(v1[valid, :], axis=1)
        / np.linalg.norm(line_vec[valid, :], axis=1)
    )

    valid[valid] = cos_a > 0.9

    return p_si, valid


if __name__ == "__main__":
    a: ArrayNx3
    a = np.array([[0, 0, 0], [0, 0, 1], [0, 0, -1], [0, 0, -1]])
    b = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0]])
    c = np.array([0, 0, 0])
    d = np.array([0, 0, 1])
    intersection_plane_mat_operation(a, b, c, d)
