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

"""Ray intersection with a bilinear patch."""

import numpy as np
from numpy.polynomial import Polynomial

from weitsicht.utils import ArrayNx3, Vector3D

__all__ = ["multilinear_poly_intersection"]


def multilinear_poly_intersection(points: ArrayNx3, p: Vector3D, r: Vector3D) -> Vector3D | None:
    """Calculate the intersection point of a ray with a bilinear polynom.
    The ray is defined by a point p and a direction r. The bilinear patch is defined by 4 points
    It is assumed that the points form a rectangle as this is used to test the validity of intersection point
    to be within the rectangle.

    :param points: The 4 corner points of the bilinear patch.
    :type points: ArrayNx3
    :param p: A point on the ray. The point location is important as for
                2 intersection solutions the closer solution to p will be used
    :type p: Vector3D
    :param r: The direction vector
    :type r: Vector3D
    :return: The first intersection point or None if no intersection is found within the limits of the points
    :rtype: Vector3D | None
    """

    # z = a00 + a01 * x + a10 * y + a11 * x * y
    # ray = p + r * t
    # z_ray = p_z + r_z * t
    # x_ray = p_x + r_x * t
    # y_ray = p_y + r_y * t

    # substitute z with z_ray and x,y of the bilinear polynom with x_ray and y_ray
    # will give us the quadratic equation for t

    # First we will solve the linear equation to get a00, a01, a10, a11
    lin_matrix = np.array([[1, 1, 1, 1], points[:, 0], points[:, 1], points[:, 0] * points[:, 1]]).T
    coeff = np.linalg.solve(lin_matrix, points[:, 2])
    a00, a01, a10, a11 = coeff

    # Second we will create the polynom of order 2
    poly = Polynomial(
        [
            -p[2] + a00 + a01 * p[0] + a10 * p[1] + a11 * p[0] * p[1],
            -r[2] + a01 * r[0] + a10 * r[1] + a11 * p[0] * r[1] + a11 * p[1] * r[0],
            a11 * r[0] * r[1],
        ]
    )

    # Finding the roots gives the value of t for the intersection points.
    roots = poly.roots()

    roots = roots[np.isreal(roots)]
    if len(roots) == 0:
        return None

    # Just to be sure. Maybe there are numerical instabilities
    roots = np.real(roots)
    p_solutions = p + np.outer(r, roots).T

    # points[:,:2].min(axis=0)<= p_solutions[:,:2]
    x_within = np.logical_and(
        (points[:, 0].min() - 1e-10) <= p_solutions[:, 0],
        p_solutions[:, 0] <= (points[:, 0].max() + 1e-10),
    )

    y_within = np.logical_and(
        (points[:, 1].min() - 1e-10) <= p_solutions[:, 1],
        p_solutions[:, 1] <= (points[:, 1].max() + 1e-10),
    )

    valid_solution_index = np.flatnonzero(np.logical_and(x_within, y_within))

    if valid_solution_index.size == 0:
        return None

    elif valid_solution_index.size == 1:
        return p_solutions[valid_solution_index, :][0]

    # More than one solution
    # Will use the closer on
    else:
        close_index = np.argmin(np.linalg.norm(p - p_solutions, axis=1))
        return p_solutions[close_index, :]
