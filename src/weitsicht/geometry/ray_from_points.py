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

"""Fit ray directions from point sets using SVD."""

import numpy as np

from weitsicht.utils import ArrayNx3, to_array_nx3

__all__ = ["ray_from_points", "rays_from_points_batch"]


def ray_from_points(points: ArrayNx3, expected_dir: ArrayNx3 | None = None, use_first_segment=True):
    """Estimate a ray (origin and direction) from a set of 3D points.

    The ray origin is the centroid of the points. The direction is the principal
    component (first right-singular vector) of the centered coordinates. The sign
    of the direction can be disambiguated via ``expected_dir`` or (if enabled)
    the first segment between the first two points.

    :param points: Input points, shape ``(N, 3)``.
    :type points: ArrayNx3
    :param expected_dir: Optional direction hint used to fix the sign, defaults to ``None``.
    :type expected_dir: array-like | None
    :param use_first_segment: Whether to use the first point segment as sign hint, defaults to ``True``.
    :type use_first_segment: bool
    :return: Tuple ``(origin, direction)`` each of shape ``(3,)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    :raises ValueError: If input shapes are incompatible.
    """

    p = to_array_nx3(points)
    c = np.mean(p, axis=0)
    q = p - c
    _, _, vh = np.linalg.svd(q, full_matrices=False)
    d = vh[0]  # principal direction, unit length

    # Pick a reference direction to avoid arbitrary sign flips.
    ref_dir = None
    if expected_dir is not None:
        ref_dir = np.asarray(expected_dir, dtype=float)
    elif use_first_segment and len(p) >= 2:
        ref_dir = p[1] - p[0]

    if ref_dir is not None:
        if np.dot(d, ref_dir) < 0:
            d = -d
    return c, d


def rays_from_points_batch(points_batch, expected_dirs: ArrayNx3 | None = None, use_first_segment=True):
    """Compute ray origins and directions for multiple point sets.

    :param points_batch: Array-like with shape ``(n_sets, n_points, 3)``.
    :type points_batch: array-like
    :param expected_dirs: Optional array-like with shape ``(n_sets, 3)`` providing sign hints per set,
        defaults to ``None``.
    :type expected_dirs: ArrayNx3 | None
    :param use_first_segment: Whether to use the first segment as sign hint, defaults to ``True``.
    :type use_first_segment: bool
    :return: Tuple ``(origins, directions)`` each shaped ``(n_sets, 3)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    :raises ValueError: If input shapes are incompatible.
    """
    p_batch = np.asarray(points_batch, dtype=float)
    if p_batch.ndim != 3 or p_batch.shape[2] != 3:
        raise ValueError("points_batch must have shape (n_sets, n_points, 3)")

    n_sets = p_batch.shape[0]

    exp = None
    if expected_dirs is not None:
        exp = np.asarray(expected_dirs, dtype=float)
        if exp.shape != (n_sets, 3):
            raise ValueError("expected_dirs must have shape (n_sets, 3)")

    # Center each set
    origins = p_batch.mean(axis=1)
    q_batch = p_batch - origins[:, None, :]

    # Batched SVD to get principal direction (first right-singular vector)
    _, _, vh = np.linalg.svd(q_batch, full_matrices=False)
    directions = vh[:, 0, :]  # shape (n_sets, 3)

    # Build reference directions for sign disambiguation
    ref = None
    if exp is not None:
        ref = exp
    elif use_first_segment and p_batch.shape[1] >= 2:
        ref = p_batch[:, 1, :] - p_batch[:, 0, :]

    if ref is not None:
        dots = np.sum(directions * ref, axis=1, keepdims=True)
        signs = np.where(dots < 0, -1.0, 1.0)
        directions = directions * signs

    return origins, directions
