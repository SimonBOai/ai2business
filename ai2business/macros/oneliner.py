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
"""Here all oneliner functions listed.

Oneliner a functions, which bundels several different modules from `AI2Business`.
"""
from ai2business.kpi_collector import trends_collector


class TrendSearch:
    """Here is collection of *oneliner* for trend search.

    !!! example "What is an *oneliner*?"

        Oneliner is a function that can be executed in one line and contains all essential
        parameters. No further introductions are needed.

    ## A four step search is searching:

        1. Overtime
        2. By regions
        3. By related topics
        4. By related queries
    """

    def four_step_search(
        keyword_list: list,
        timeframe: str = "today 5-y",
        language: str = "en-US",
        category: int = 0,
        timezone: int = 360,
        country: str = "",
        property_filter="",
        resolution: str = "COUNTRY",
        **kwargs,
    ) -> dict:
        """four step search

        A four step search is searching:

        1. Overtime
        2. By regions
        3. By related topics
        4. By related queries

        Args:
            keyword_list (list): Keyword-list with the items to search for.
            timeframe (str, optional): Time frame, respectively, period to search for.
            Defaults to "today 5-y".
            language (str, optional): Search language. Defaults to "en-US".
            category (int, optional): Define a specific [search category](https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories). Defaults to 0.
            timezone (int, optional): [Search timezone](https://developers.google.com/maps/documentation/timezone/overview). Defaults to 360.
            country (str, optional): The country, where to search for. Defaults to "".
            property_filter (str, optional): Property filer of the search; only in news, images, YouTube, shopping. Defaults to "".
            resolution (str, optional): The resolution / zoom-factor, where to search for. Defaults to "COUNTRY".

        Returns:
            dict: Dictionary with the keys: 'interest_over_time', 'interest_by_region', 'related_topics', 'related_queries' and its dataframes.
        """
        trends = trends_collector.TrendsCollector()
        builder = trends_collector.DesignerTrendsCollector(
            keyword_list=keyword_list,
            timeframe=timeframe,
            language=language,
            category=category,
            timezone=timezone,
            country=country,
            property_filter=property_filter,
            **kwargs,
        )
        trends.builder = builder
        trends.find_interest_over_time()
        trends.find_interest_by_region(resolution=resolution)
        trends.find_related_topics()
        trends.find_related_queries()
        return builder.trends.return_product
