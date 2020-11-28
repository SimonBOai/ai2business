import pandas as pd

from ai2business.kpi_collector import trends_collector as tdc

trends = tdc.TrendsCollector()
builder = tdc.DesignerTrendsCollector(["AI", "Business", "AI2Business"])
trends.builder = builder


def test_dataframes_1():
    trends.find_interest_over_time()
    assert type(builder.return_dataframe) == type(pd.DataFrame())


def test_dataframes_2():
    trends.find_interest_over_time()
    assert type(builder.return_dataframe) == type(pd.DataFrame())


def test_dataframes_3():
    trends.find_interest_over_time()
    assert type(builder.return_dataframe) == type(pd.DataFrame())


def test_dataframes_4():
    trends.find_trending_searches()
    assert type(builder.return_dataframe) == type(pd.DataFrame())


def test_dataseries_5():
    trends.find_today_searches()
    assert type(builder.return_dataframe) == type(pd.Series(dtype=object))


def test_dataframes_6():
    trends.find_top_charts(2018)
    assert type(builder.return_dataframe) == type(pd.DataFrame())


def test_dataframes_6_failed():
    trends.find_top_charts(2020)
    assert type(builder.return_dataframe) == type(pd.DataFrame())


def test_datadict_7():
    trends.find_related_topics()
    assert type(builder.return_dict) == type(dict())


def test_datadict_8():
    trends.find_related_queries()
    assert type(builder.return_dict) == type(dict())


def test_datadict_9():
    trends.find_suggestions()
    assert type(builder.return_dict) == type(dict())


def test_datadict_10():
    trends.find_categories()
    assert type(builder.return_dict) == type(dict())


def test_dataframes_11():
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


def test_dataframes_11_failed():
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
