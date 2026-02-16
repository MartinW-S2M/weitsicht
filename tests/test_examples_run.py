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

import runpy
from pathlib import Path

import pytest

# Collect all top-level example scripts
EXAMPLES = sorted(Path(__file__).resolve().parent.parent.joinpath("examples").glob("*.py"))


@pytest.mark.slow
@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_script_runs(script, monkeypatch):
    # Ensure relative data paths in examples keep working
    monkeypatch.chdir(script.parent)
    # Execute the example as a script; any exception will fail the test
    runpy.run_path(str(script), run_name="__main__")
