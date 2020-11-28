from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty

import pandas as pd
import yfinance as yf
from yfinance.utils import parse_actions

class BuilderFinanceCollector(ABC):
    @abstractproperty
    def return_dataframe(self) -> None:
        pass
    
    @abstractmethod
    def get_chart_history(self) -> None:
        pass
    
    @abstractmethod
    def get_isin_code(self) -> None:
        pass
    
    @abstractmethod
    def get_major_holders(self) -> None:
        pass
    
    @abstractmethod
    def get_mutualfund_holders(self) -> None:
        pass
    
    @abstractmethod
    def get_dividends(self) -> None:
        pass
    
    @abstractmethod
    def get_splits(self) -> None:
        pass
    
    @abstractmethod
    def get_actions(self) -> None:
        pass
    
    @abstractmethod
    def get_info(self) -> None:
        pass
    
    @abstractmethod
    def get_calendar(self) -> None:
        pass
    
    @abstractmethod
    def get_recommendations(self) -> None:
        pass
    
    @abstractmethod
    def get_earnings(self) -> None:
        pass
    
    @abstractmethod
    def get_quarterly_eanrings(self) -> None:
        pass
    
    @abstractmethod
    def get_financials(self) -> None:
        pass
    @abstractmethod
    def get_quarterly_financials(self) -> None:
        pass
    
    @abstractmethod
    def get_balance_sheet(self) -> None:
        pass
    
    @abstractmethod
    def get_quarterly_balance_sheet(self) -> None:
        pass
    
    @abstractmethod
    def get_cashflow(self) -> None:
        pass
    
    @abstractmethod
    def get_quarterly_cashflow(self) -> None:
        pass
    
    @abstractmethod
    def get_sustainability(self) -> None:
        pass
    
    @abstractmethod
    def get_options(self) -> None:
        pass
    
 class DesignerFinanceCollector(BuilderFinanceCollector):
     def __init__(self) -> None:
         pass
    
    
    