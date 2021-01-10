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
"""Theme mode for seaborn."""
import seaborn as sns


def light_mode(grid: bool = False) -> sns.set_theme:
    """Set the seaborn theme style to light mode.

    Args:
        grid (bool, optional): Activate grid mode for plot. Defaults to False.

    Returns:
        sns.set_theme: Environment setting function of seaborn.
    """
    if grid:
        return sns.set_theme(style="whitegrid")
    sns.set_theme(style="white")


def dark_mode(grid: bool = False) -> sns.set_theme:
    """Set the seaborn theme style to dark mode.

    Args:
        grid (bool, optional): Activate grid mode for plot. Defaults to False.

    Returns:
        sns.set_theme: Environment setting function of seaborn.
    """
    if grid:
        return sns.set_theme(style="darkgrid")
    return sns.set_theme(style="dark")
