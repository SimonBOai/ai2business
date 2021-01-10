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
"""Test-Environment for trends_collector."""
import pandas as pd

from ai2business.kpi_collector import trends_collector as tdc

trends = tdc.TrendsCollector()
builder = tdc.DesignerTrendsCollector(["AI", "Business", "AI2Business"])
trends.builder = builder


def test_interest_over_time() -> None:
    trends.find_interest_over_time()
    assert type(builder.trends.return_product["get_interest_over_time"]) == type(
        pd.DataFrame()
    )


def test_interest_by_region() -> None:
    trends.find_interest_by_region()
    assert type(builder.trends.return_product["get_interest_by_region"]) == type(
        pd.DataFrame()
    )


def test_trending_searches() -> None:
    trends.find_trending_searches("japan")
    assert type(builder.trends.return_product["get_trending_searches"]) == type(
        pd.DataFrame()
    )


def test_today_searches() -> None:
    trends.find_today_searches()
    assert type(builder.trends.return_product["get_today_searches"]) == type(
        pd.Series(dtype=object)
    )


def test_top_charts_true() -> None:
    trends.find_top_charts(2018)
    assert type(builder.trends.return_product["get_top_charts"]) == type(pd.DataFrame())


def test_top_charts_failed() -> None:
    trends.find_top_charts(2022)
    assert builder.trends.return_product == {}


def test_related_topics() -> None:
    trends.find_related_topics()
    assert type(builder.trends.return_product["get_related_topics"]) == type(dict())


def test_related_queries() -> None:
    trends.find_related_queries()
    assert type(builder.trends.return_product["get_related_queries"]) == type(dict())


def test_suggestions() -> None:
    trends.find_suggestions()
    assert type(builder.trends.return_product["get_suggestions"]) == type(dict())


def test_categories() -> None:
    trends.find_categories()
    assert type(builder.trends.return_product["get_categories"]) == type(dict())


def test_historical_interest_true() -> None:
    trends.find_historical_interest(
        year_start=2018,
        month_start=1,
        day_start=1,
        hour_start=0,
        year_end=2018,
        month_end=2,
        day_end=1,
        hour_end=0,
    )
    assert type(builder.trends.return_product["get_historical_interest"]) == type(
        pd.DataFrame()
    )


def test_historical_interest_failed() -> None:
    trends.find_historical_interest(
        year_start=2018,
        month_start=1,
        day_start=1,
        hour_start=0,
        year_end=2018,
        month_end=2,
        day_end=30,
        hour_end=0,
    )
    assert builder.trends.return_product == {}


def test_wordcloud() -> None:
    # Have to initial new to avoid time out error
    trends = tdc.TrendsCollector()
    builder = tdc.DesignerTrendsCollector(["test", "wordcloud"])
    trends.builder = builder
    trends.make_wordcloud()
    assert type(builder.trends.return_product) == type(dict())


def test_part_list() -> None:
    # Have to initial new to avoid time out error
    trends = tdc.TrendsCollector()
    builder = tdc.DesignerTrendsCollector(["test", "wordcloud"])
    trends.builder = builder
    trends.find_related_queries()
    assert builder.trends.list_product_parts == "Product parts: get_related_queries"
