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

from weitsicht.utils import ArrayNx3, Vector3D

__all__ = [
    "bilinear_patch_normal",
    "multilinear_poly_intersection",
    "multilinear_poly_intersection_with_normal",
]


def _bilinear_coeff(points: ArrayNx3) -> tuple[float, float, float, float, float, float, float]:
    points = np.asarray(points, dtype=float)
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    (x0, y0, z00), (_x0, y1, z01), (x1, _y0, z10), (_x1, _y1, z11) = points

    if x0 != _x0 or x1 != _x1 or y0 != _y0 or y1 != _y1:
        raise ValueError("points do not form a rectangle")

    dx = x1 - x0
    dy = y1 - y0
    if dx == 0.0 or dy == 0.0:
        raise ValueError("points do not form a rectangle")

    z_ref = float(points[:, 2].mean())
    a00 = z00 - z_ref
    a01 = (z10 - z00) / dx
    a10 = (z01 - z00) / dy
    a11 = (z11 - z10 - z01 + z00) / (dx * dy)
    return float(x0), float(y0), z_ref, float(a00), float(a01), float(a10), float(a11)


def _bilinear_normal_from_coeff(a01: float, a10: float, a11: float, x: float, y: float) -> Vector3D:
    dz_dx = a01 + a11 * y
    dz_dy = a10 + a11 * x

    n = np.array([-dz_dx, -dz_dy, 1.0], dtype=float)
    n /= np.linalg.norm(n)
    return n


def _real_polynomial_roots(c0: float, c1: float, c2: float) -> np.ndarray:
    scale = max(abs(c0), abs(c1), abs(c2), 1.0)
    tol = 64.0 * np.finfo(float).eps * scale

    if abs(c2) <= tol:
        if abs(c1) <= tol:
            return np.array([], dtype=float)
        return np.array([-c0 / c1], dtype=float)

    discriminant = c1 * c1 - 4.0 * c2 * c0
    discriminant_tol = 64.0 * np.finfo(float).eps * (c1 * c1 + abs(4.0 * c2 * c0) + 1.0)
    if discriminant < -discriminant_tol:
        return np.array([], dtype=float)
    if abs(discriminant) <= discriminant_tol:
        return np.array([-c1 / (2.0 * c2)], dtype=float)

    sqrt_discriminant = float(np.sqrt(discriminant))
    q = -0.5 * (c1 + np.copysign(sqrt_discriminant, c1))
    if q == 0.0:
        return np.array([(-c1 - sqrt_discriminant) / (2.0 * c2), (-c1 + sqrt_discriminant) / (2.0 * c2)])
    return np.array([q / c2, c0 / q], dtype=float)


def bilinear_patch_normal(points: ArrayNx3, point: Vector3D) -> Vector3D:
    """Compute the surface normal of the bilinear patch at a given point.

    The bilinear patch is modeled as a height field in local cell coordinates:

        ``z(u, v) - z_ref = a00 + a01*u + a10*v + a11*u*v``.

    The normal is derived from the implicit form ``F(x,y,z) = z - z(x,y) = 0``:

        ``n = grad(F) = (-dz/dx, -dz/dy, 1)``.

    :param points: The 4 corner points of the bilinear patch.
    :type points: ArrayNx3
    :param point: 3D point where the normal should be evaluated (only x/y are used).
    :type point: Vector3D
    :return: Unit normal vector (z component is positive).
    :rtype: Vector3D
    """

    x0, y0, _z_ref, _a00, a01, a10, a11 = _bilinear_coeff(points)
    return _bilinear_normal_from_coeff(
        a01=a01,
        a10=a10,
        a11=a11,
        x=float(point[0]) - x0,
        y=float(point[1]) - y0,
    )


def _multilinear_poly_intersection_from_coeff(
    points: ArrayNx3,
    p: Vector3D,
    r: Vector3D,
    x0: float,
    y0: float,
    z_ref: float,
    a00: float,
    a01: float,
    a10: float,
    a11: float,
) -> Vector3D | None:
    p_x = p[0] - x0
    p_y = p[1] - y0
    p_z = p[2] - z_ref
    c0 = -p_z + a00 + a01 * p_x + a10 * p_y + a11 * p_x * p_y
    c1 = -r[2] + a01 * r[0] + a10 * r[1] + a11 * p_x * r[1] + a11 * p_y * r[0]
    c2 = a11 * r[0] * r[1]

    # z - z_ref = a00 + a01 * u + a10 * v + a11 * u * v
    # ray = p + r * t
    # z_ray - z_ref = p_z_shifted + r_z * t
    # u_ray = p_x - x0 + r_x * t
    # v_ray = p_y - y0 + r_y * t

    # substitute z with z_ray and u,v of the bilinear polynom with u_ray and v_ray
    # will give us the quadratic equation for t
    # Finding the roots gives the value of t for the intersection points.

    roots = _real_polynomial_roots(c0=c0, c1=c1, c2=c2)
    if len(roots) == 0:
        return None

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

    # z - z_ref = a00 + a01 * u + a10 * v + a11 * u * v
    # ray = p + r * t
    # z_ray - z_ref = p_z_shifted + r_z * t
    # u_ray = p_x - x0 + r_x * t
    # v_ray = p_y - y0 + r_y * t

    # substitute z with z_ray and u,v of the bilinear polynom with u_ray and v_ray
    # will give us the quadratic equation for t
    # Finding the roots gives the value of t for the intersection points.

    x0, y0, z_ref, a00, a01, a10, a11 = _bilinear_coeff(points)
    intersect = _multilinear_poly_intersection_from_coeff(
        points=points,
        p=p,
        r=r,
        x0=x0,
        y0=y0,
        z_ref=z_ref,
        a00=a00,
        a01=a01,
        a10=a10,
        a11=a11,
    )
    if intersect is None:
        return None

    normal = _bilinear_normal_from_coeff(
        a01=a01,
        a10=a10,
        a11=a11,
        x=float(intersect[0]) - x0,
        y=float(intersect[1]) - y0,
    )
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

    # z - z_ref = a00 + a01 * u + a10 * v + a11 * u * v
    # ray = p + r * t
    # z_ray - z_ref = p_z_shifted + r_z * t
    # u_ray = p_x - x0 + r_x * t
    # v_ray = p_y - y0 + r_y * t

    # substitute z with z_ray and u,v of the bilinear polynom with u_ray and v_ray
    # will give us the quadratic equation for t

    x0, y0, z_ref, a00, a01, a10, a11 = _bilinear_coeff(points)
    return _multilinear_poly_intersection_from_coeff(
        points=points,
        p=p,
        r=r,
        x0=x0,
        y0=y0,
        z_ref=z_ref,
        a00=a00,
        a01=a01,
        a10=a10,
        a11=a11,
    )
