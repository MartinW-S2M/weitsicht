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

"""Bilinear interpolation utilities."""

__all__ = ["bilinear_interpolation"]


def bilinear_interpolation(x: float | int, y: float | int, points: tuple | list) -> float:
    """Bilinear interpolation on a rectangle defined by four corner samples.

    The ``points`` must be a sequence of four ``(x, y, value)`` tuples forming a rectangle.

    :param x: X coordinate where the value should be interpolated.
    :type x: float | int
    :param y: Y coordinate where the value should be interpolated.
    :type y: float | int
    :param points: Four corner points ``(x, y, value)``.
    :type points: tuple | list
    :return: Interpolated value.
    :rtype: float
    :raises ValueError: If the points do not form a rectangle or if ``(x, y)`` lies outside the rectangle.
    :raises TypeError: If values have incompatible types.



        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    """
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)  # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError("points do not form a rectangle")
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError("(x, y) not within the rectangle")

    return (
        q11 * (x2 - x) * (y2 - y) + q21 * (x - x1) * (y2 - y) + q12 * (x2 - x) * (y - y1) + q22 * (x - x1) * (y - y1)
    ) / ((x2 - x1) * (y2 - y1))
