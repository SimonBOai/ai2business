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
"""Test-Environment for oneliner."""

from ai2business.macros import oneliner


def test_four_step_search() -> None:

    result = oneliner.TrendSearch.four_step_search(keyword_list=["2019", "2020"])
    assert list(result.keys()) == [
        "get_interest_over_time",
        "get_interest_by_region",
        "get_related_topics",
        "get_related_queries",
    ]
