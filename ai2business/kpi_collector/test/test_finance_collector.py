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
"""Test-Environment for finance_collector."""
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


def test_find_major_holders_dict():
    ticker.find_major_holders()
    assert type(builder.return_dict) == type(dict())


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


def test_find_recommendation():
    ticker.find_recommendations()
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


def test_find_cashflow():
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
