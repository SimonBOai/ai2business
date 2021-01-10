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
"""Generates sample list of KPIs"""
from ai2business.datasets.data import database


class SampleGenerators:
    """Sample Generators allows to generate key word list.

    !!! example
        The module `sample_generator.py` contains functions, which allows generating a list of
        keywords with and without acronym.

    ```python
    # Get ticker values of the leading stock markert worldwide.
    stock_market(indices: str = "DOWJONES") -> dict
    ```
    """

    pass


def stock_market(indices: str = "DOWJONES") -> dict:
    """Returns all company names and ISIN for a given stock market.

    Args:
        indices (str, optional): Name of the stock market. Defaults to "DOWJONES".

    Returns:
        dict: Collection of the indices of the current stock market.
    """
    try:
        return database.StockMarket.__dict__[indices.lower()]
    except KeyError as exc:
        print(f"ERROR: {exc} -> Indices is not listed in the database!")
        return {}
