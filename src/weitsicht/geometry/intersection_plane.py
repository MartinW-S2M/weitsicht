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
    plane_point: Vector3D | ArrayNx3,
    plane_normal: Vector3D | ArrayNx3 | None = None,
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
    :type plane_point: Vector3D | ArrayNx3
    :param plane_normal: Plane normal vector, defaults to ``None`` (interpreted as ``(0, 0, 1)``).
        Can be a single normal for all rays (shape ``(3,)``) or per-ray normals (shape ``(N, 3)``).
    :type plane_normal: Vector3D | ArrayNx3 | None
    :return: Tuple ``(intersection_points, valid_mask)``. Invalid intersections contain NaNs.
    :rtype: tuple[ArrayNx3, ``MaskN_``]
    """
    _line_vec = np.asarray(line_vec, dtype=float)
    _line_point = np.asarray(line_point, dtype=float)

    if _line_vec.ndim != 2 or _line_vec.shape[1] != 3:
        raise ValueError("line_vec must have shape (N, 3)")
    if _line_point.shape != _line_vec.shape:
        raise ValueError("line_point must have the same shape as line_vec")

    n_rays = _line_vec.shape[0]

    epsilon = 1e-7
    # Plane point (broadcast to per-ray shape if needed).
    _plane_point = np.asarray(plane_point, dtype=float)
    if _plane_point.ndim == 1:
        if _plane_point.shape != (3,):
            raise ValueError("plane_point must have shape (3,) or (N, 3)")
        _plane_point = np.broadcast_to(_plane_point, _line_point.shape)
    elif _plane_point.ndim == 2:
        if _plane_point.shape != _line_point.shape:
            raise ValueError("plane_point must have shape (3,) or (N, 3) matching line_vec/line_point")
    else:
        raise ValueError("plane_point must have shape (3,) or (N, 3)")

    # Plane normal: can be a single normal for all rays or per-ray normals.
    if plane_normal is None:
        pln = np.array([0.0, 0.0, 1.0], dtype=float)
        pln_per_ray = False
    else:
        pln = np.asarray(plane_normal, dtype=float)
        if pln.ndim == 1:
            if pln.shape != (3,):
                raise ValueError("plane_normal must have shape (3,) or (N, 3)")
            pln_per_ray = False
        elif pln.ndim == 2:
            if pln.shape != _line_vec.shape:
                raise ValueError("plane_normal must have shape (3,) or (N, 3) matching line_vec/line_point")
            pln_per_ray = True
        else:
            raise ValueError("plane_normal must have shape (3,) or (N, 3)")

    if pln_per_ray:
        n_dotu = np.einsum("ij,ij->i", _line_vec, pln)
    else:
        n_dotu = _line_vec.dot(pln)

    # check for parallel lines to plane
    valid = abs(n_dotu) > epsilon

    w = _line_point - _plane_point

    si = np.zeros((n_rays,), dtype=float)
    if np.any(valid):
        if pln_per_ray:
            numer = np.einsum("ij,ij->i", -w[valid, :], pln[valid, :])
        else:
            numer = np.dot(-w[valid, :], pln)
        si[valid] = numer / n_dotu[valid]

    p_si = np.empty((_line_point.shape[0], 3), dtype=float)
    p_si.fill(np.nan)
    if np.any(valid):
        p_si[valid, :] = _line_point[valid, :] + _line_vec[valid, :] * si[valid, None]

    # Check if the direction would be to the other direction aka backwards
    # because that is mathematically possible on a line plane intersection
    v1 = np.zeros(w.shape, dtype=float)
    v1[valid, :] = p_si[valid, :] - _line_point[valid, :]
    valid[valid] = np.logical_and(valid[valid], np.linalg.norm(v1[valid, :], axis=1) > epsilon)
    cos_a = (
        np.einsum("ij,ij->i", v1[valid, :], _line_vec[valid, :])
        / np.linalg.norm(v1[valid, :], axis=1)
        / np.linalg.norm(_line_vec[valid, :], axis=1)
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
