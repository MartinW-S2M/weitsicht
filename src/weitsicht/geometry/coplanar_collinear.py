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

"""3D coplanarity checks."""

__all__ = ["is_coplanar"]

# Python program to check if 4 points
# in a 3-D plane are Coplanar


# Function to find equation of plane.
def is_coplanar(p1, p2, p3, p_test):
    """Return whether four 3D points are coplanar.

    Points are interpreted as array-like objects of length 3: ``(x, y, z)``.

    :param p1: First point.
    :type p1: array-like
    :param p2: Second point.
    :type p2: array-like
    :param p3: Third point.
    :type p3: array-like
    :param p_test: Fourth point to test.
    :type p_test: array-like
    :return: ``True`` if the four points are coplanar, otherwise ``False``.
    :rtype: bool
    """
    a1 = p2[0] - p1[0]
    b1 = p2[1] - p1[1]
    c1 = p2[2] - p1[2]
    a2 = p3[0] - p1[0]
    b2 = p3[1] - p1[1]
    c2 = p3[2] - p1[2]
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = -a * p1[0] - b * p1[1] - c * p1[2]

    # equation of plane is: a*x + b*y + c*z = 0 #

    # checking if the 4th point satisfies
    # the above equation
    if a * p_test[0] + b * p_test[1] + c * p_test[2] + d == 0:
        return True
    else:
        return False
