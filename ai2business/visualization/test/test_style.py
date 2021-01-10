# Copyright 2020 AI2Business. All Rights Reserved.
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
# ==============================================================================
"""Test-Environment for seaborn figure style."""

import seaborn as sns

from ai2business.visualization import style


def test_light_mode():

    assert style.light_mode() == sns.set_theme(style="white")


def test_light_mode_grid():

    assert style.light_mode(grid=True) == sns.set_theme(style="whitegrid")


def test_dark_mode():

    assert style.dark_mode() == sns.set_theme(style="dark")


def test_dark_mode_grid():

    assert style.dark_mode(grid=True) == sns.set_theme(style="darkgrid")
