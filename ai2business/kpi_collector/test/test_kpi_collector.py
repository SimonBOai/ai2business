from ai2business.kpi_collector import trends_collector as tdc
import pandas as pd
import pytest

@pytest.mark.incremental
def test_1():
    trends = tdc.TrendsCollector()
    builder = tdc.DesignerTrendsCollector()
    trends.builder = builder
    trends.find_interest_over_time()
    print(type(builder.return_dataframe) == type(pd.DataFrame()))