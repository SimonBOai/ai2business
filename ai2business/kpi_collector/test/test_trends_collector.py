import pandas as pd

from ai2business.kpi_collector import trends_collector as tdc

trends = tdc.TrendsCollector()
builder = tdc.DesignerTrendsCollector(["AI", "Business", "AI2Business"])
trends.builder = builder


def test_interest_over_time() -> None:
    trends.find_interest_over_time()
    assert type(builder.return_dataframe) == type(pd.DataFrame())


def test_interest_by_region() -> None:
    trends.find_interest_by_region()
    assert type(builder.return_dataframe) == type(pd.DataFrame())


def test_trending_searches() -> None:
    trends.find_trending_searches()
    assert type(builder.return_dataframe) == type(pd.DataFrame())


def test_today_searches() -> None:
    trends.find_today_searches()
    assert type(builder.return_dataframe) == type(pd.Series(dtype=object))


def test_top_charts_true() -> None:
    trends.find_top_charts(2018)
    assert type(builder.return_dataframe) == type(pd.DataFrame())


def test_top_charts_failed() -> None:
    trends.find_top_charts(2020)
    assert type(builder.return_dataframe) == type(pd.DataFrame())


def test_related_topics() -> None:
    trends.find_related_topics()
    assert type(builder.return_dict) == type(dict())


def test_related_queries() -> None:
    trends.find_related_queries()
    assert type(builder.return_dict) == type(dict())


def test_suggestions() -> None:
    trends.find_suggestions()
    assert type(builder.return_dict) == type(dict())


def test_categories() -> None:
    trends.find_categories()
    assert type(builder.return_dict) == type(dict())


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
    assert type(builder.return_dataframe) == type(pd.DataFrame())


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
    assert type(builder.return_dataframe) == type(pd.DataFrame())
