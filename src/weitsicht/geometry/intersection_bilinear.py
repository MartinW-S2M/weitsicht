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

__all__ = [
    "bilinear_patch_normal",
    "multilinear_poly_intersection",
    "multilinear_poly_intersection_with_normal",
]


def _bilinear_coeff(points: ArrayNx3) -> tuple[float, float, float, float]:
    lin_matrix = np.array([[1, 1, 1, 1], points[:, 0], points[:, 1], points[:, 0] * points[:, 1]]).T
    a00, a01, a10, a11 = np.linalg.solve(lin_matrix, points[:, 2])
    return float(a00), float(a01), float(a10), float(a11)


def _bilinear_normal_from_coeff(a01: float, a10: float, a11: float, x: float, y: float) -> Vector3D:
    dz_dx = a01 + a11 * y
    dz_dy = a10 + a11 * x

    n = np.array([-dz_dx, -dz_dy, 1.0], dtype=float)
    n /= np.linalg.norm(n)
    return n


def bilinear_patch_normal(points: ArrayNx3, point: Vector3D) -> Vector3D:
    """Compute the surface normal of the bilinear patch at a given point.

    The bilinear patch is modeled as a height field:

        ``z(x, y) = a00 + a01*x + a10*y + a11*x*y``.

    The normal is derived from the implicit form ``F(x,y,z) = z - z(x,y) = 0``:

        ``n = grad(F) = (-dz/dx, -dz/dy, 1)``.

    :param points: The 4 corner points of the bilinear patch.
    :type points: ArrayNx3
    :param point: 3D point where the normal should be evaluated (only x/y are used).
    :type point: Vector3D
    :return: Unit normal vector (z component is positive).
    :rtype: Vector3D
    """

    _a00, a01, a10, a11 = _bilinear_coeff(points)
    return _bilinear_normal_from_coeff(a01=a01, a10=a10, a11=a11, x=float(point[0]), y=float(point[1]))


def _multilinear_poly_intersection_from_coeff(
    points: ArrayNx3,
    p: Vector3D,
    r: Vector3D,
    a00: float,
    a01: float,
    a10: float,
    a11: float,
) -> Vector3D | None:
    poly = Polynomial(
        [
            -p[2] + a00 + a01 * p[0] + a10 * p[1] + a11 * p[0] * p[1],
            -r[2] + a01 * r[0] + a10 * r[1] + a11 * p[0] * r[1] + a11 * p[1] * r[0],
            a11 * r[0] * r[1],
        ]
    )

    # z = a00 + a01 * x + a10 * y + a11 * x * y
    # ray = p + r * t
    # z_ray = p_z + r_z * t
    # x_ray = p_x + r_x * t
    # y_ray = p_y + r_y * t

    # substitute z with z_ray and x,y of the bilinear polynom with x_ray and y_ray
    # will give us the quadratic equation for t
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

    if valid_solution_index.size == 1:
        return p_solutions[valid_solution_index, :][0]

    # More than one solution
    # Will use the closer on
    close_index = np.argmin(np.linalg.norm(p - p_solutions, axis=1))
    return p_solutions[close_index, :]


def multilinear_poly_intersection_with_normal(
    points: ArrayNx3,
    p: Vector3D,
    r: Vector3D,
    *,
    orient_normal_to_ray: bool = True,
) -> tuple[Vector3D, Vector3D] | None:
    """Calculate ray intersection point and surface normal for a bilinear patch.

    :param points: The 4 corner points of the bilinear patch.
    :type points: ArrayNx3
    :param p: A point on the ray.
    :type p: Vector3D
    :param r: The direction vector of the ray.
    :type r: Vector3D
    :param orient_normal_to_ray: If True, flips the normal so that ``normal dot r <= 0``.
    :type orient_normal_to_ray: bool
    :return: Tuple ``(intersection_point, unit_normal)`` or ``None`` if no intersection is found.
    :rtype: tuple[Vector3D, Vector3D] | None
    """

    # z = a00 + a01 * x + a10 * y + a11 * x * y
    # ray = p + r * t
    # z_ray = p_z + r_z * t
    # x_ray = p_x + r_x * t
    # y_ray = p_y + r_y * t

    # substitute z with z_ray and x,y of the bilinear polynom with x_ray and y_ray
    # will give us the quadratic equation for t
    # Finding the roots gives the value of t for the intersection points.

    a00, a01, a10, a11 = _bilinear_coeff(points)
    intersect = _multilinear_poly_intersection_from_coeff(points=points, p=p, r=r, a00=a00, a01=a01, a10=a10, a11=a11)
    if intersect is None:
        return None

    normal = _bilinear_normal_from_coeff(a01=a01, a10=a10, a11=a11, x=float(intersect[0]), y=float(intersect[1]))
    if orient_normal_to_ray and float(np.dot(normal, r)) > 0.0:
        normal = -normal

    return intersect, normal


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

    a00, a01, a10, a11 = _bilinear_coeff(points)
    return _multilinear_poly_intersection_from_coeff(points=points, p=p, r=r, a00=a00, a01=a01, a10=a10, a11=a11)
