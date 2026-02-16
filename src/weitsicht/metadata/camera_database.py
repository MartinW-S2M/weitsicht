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

# Todo advanced database with model and make similarity search

#: Sensor (width_mm, height_mm) keyed by camera model name.
ImageSize: dict[str, tuple[float, float]]
ImageSize = {
    # ... existing entries ..
    "L1D-20c": (13.20, 8.80),
    "L2D-20c": (13.20, 8.80),
    "FC3411": (13.2, 8.8),
    "DSC-RX1RM2": (35.9, 24.00),
    "ILCE-1": (35.9, 24),
    "Canon5DSR": (36, 24),
    "GFX100S": (43.8, 32.9),
    "GFX100 II": (43.8, 32.9),
    "NIKON D3200": (23.2, 15.4),
}


def get_sensor_from_database(make, model) -> tuple | None:
    # Find sensor in database
    sensor_width = None
    if model in ImageSize:
        sensor_width = ImageSize[model]

    return sensor_width
