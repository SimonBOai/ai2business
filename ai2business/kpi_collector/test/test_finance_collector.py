import pandas as pd

from ai2business.kpi_collector import finance_collector as fnc

ticker = fnc.FinanceCollector()
builder = fnc.DesignerFinanceCollector(["SPY", "AAPL", "MSFT"])
ticker.builder = builder


def test_find_chart_history():
    ticker.find_chart_histogram()
    assert type(builder.return_dataframe) == type(pd.DataFrame())


def test_find_isin_code():
    ticker.find_isin_code()
    assert type(builder.return_dict) == type(dict())


def test_dict_as_dataframe_true():
    ticker.find_isin_code()
    assert type(builder.return_dict) == type(dict())


def test_dict_as_dataframe_false():
    ticker.find_isin_code()
    assert builder.return_dict_as_dataframe == None


def test_find_major_holders_dict():
    ticker.find_major_holders()
    assert type(builder.return_dict) == type(dict())


def test_find_major_holders_df():
    ticker.find_major_holders()
    assert type(builder.return_dict_as_dataframe) == type(pd.DataFrame())


def test_find_institutional_holders():
    ticker.find_institutional_holders()
    assert type(builder.return_dict) == type(dict())


def test_find_mutualfund_holders():
    ticker.find_mutualfund_holders()
    assert type(builder.return_dict) == type(dict())


def test_find_dividends():
    ticker.find_dividends()
    assert type(builder.return_dict) == type(dict())


def test_find_splits():
    ticker.find_splits()
    assert type(builder.return_dict) == type(dict())


def test_find_actions():
    ticker.find_actions()
    assert type(builder.return_dict) == type(dict())


def test_find_info():
    ticker.find_info()
    assert type(builder.return_dict) == type(dict())


def test_find_calendar():
    ticker.find_calendar()
    assert type(builder.return_dict) == type(dict())


def test_find_earnings():
    ticker.find_earnings()
    assert type(builder.return_dict) == type(dict())


def test_find_quarterly_earnings():
    ticker.find_quarterly_earnings()
    assert type(builder.return_dict) == type(dict())


def test_find_financials():
    ticker.find_financials()
    assert type(builder.return_dict) == type(dict())


def test_find_quarterly_financials():
    ticker.find_quarterly_financials()
    assert type(builder.return_dict) == type(dict())


def test_find_balancesheet():
    ticker.find_balancesheet()
    assert type(builder.return_dict) == type(dict())


def test_find_quarterly_balancesheet():
    ticker.find_quarterly_balancesheet()
    assert type(builder.return_dict) == type(dict())


def test_find_cashflows():
    ticker.find_cashflow()
    assert type(builder.return_dict) == type(dict())


def test_find_quarterly_cashflow():
    ticker.find_quarterly_cashflow()
    assert type(builder.return_dict) == type(dict())


def test_find_sustainability():
    ticker.find_sustainability()
    assert type(builder.return_dict) == type(dict())


def test_find_options():
    ticker.find_options()
    assert type(builder.return_dict) == type(dict())
