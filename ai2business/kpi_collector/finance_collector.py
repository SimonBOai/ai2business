""" [summary]

[extended_summary]

"""
from abc import ABC, abstractmethod, abstractproperty, abstractstaticmethod

import pandas as pd
import yfinance as yf


class BuilderFinanceCollector(ABC):
    """BuilderFinanceCollector [summary]

    [extended_summary]

    Args:
        ABC ([type]): [description]
    """

    @abstractproperty
    def return_dataframe(self) -> None:
        """return_dataframe [summary]

        [extended_summary]
        """
        pass

    @abstractproperty
    def return_dict(self) -> None:
        """return_dict [summary]

        [extended_summary]
        """
        pass

    @abstractstaticmethod
    def all_trickers() -> None:
        """all_trickers [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_chart_history(self) -> None:
        """get_chart_history [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_isin_code(self) -> None:
        """get_isin_code [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_major_holders(self) -> None:
        """get_major_holders [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_institutional_holders(self) -> None:
        """get_institutional_holders [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_mutualfund_holders(self) -> None:
        """get_mutualfund_holders [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_dividends(self) -> None:
        """get_dividends [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_splits(self) -> None:
        """get_splits [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_actions(self) -> None:
        """get_actions [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_info(self) -> None:
        """get_info [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_calendar(self) -> None:
        """get_calendar [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_recommendations(self) -> None:
        """get_recommendations [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_earnings(self) -> None:
        """get_earnings [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_quarterly_earnings(self) -> None:
        """get_quarterly_earnings [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_financials(self) -> None:
        """get_financials [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_quarterly_financials(self) -> None:
        """get_quarterly_financials [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_balancesheet(self) -> None:
        """get_balancesheet [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_quarterly_balancesheet(self) -> None:
        """get_quarterly_balancesheet [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_cashflow(self) -> None:
        """get_cashflow [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_quarterly_cashflow(self) -> None:
        """get_quarterly_cashflow [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_sustainability(self) -> None:
        """get_sustainability [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_options(self) -> None:
        """get_options [summary]

        [extended_summary]
        """
        pass


class DesignerFinanceCollector(BuilderFinanceCollector):
    """DesignerFinanceCollector [summary]

    [extended_summary]

    Args:
        BuilderFinanceCollector ([type]): [description]
    """

    def __init__(self, key_word_list: list) -> None:
        """__init__ [summary]

        [extended_summary]

        Args:
            key_word_list (list): [description]
        """
        self.key_word_list = key_word_list
        self.tickers = yf.Tickers(" ".join(self.key_word_list))
        self.df = pd.DataFrame()
        self.dict = {}

    @property
    def return_dataframe(self) -> pd.DataFrame:
        """return_dataframe [summary]

        [extended_summary]

        Returns:
            pd.DataFrame: [description]
        """
        return self.df

    @property
    def return_dict(self) -> dict:
        """return_dict [summary]

        [extended_summary]

        Returns:
            dict: [description]
        """
        return self.dict

    @staticmethod
    def all_trickers(
        tickers: yf.Tickers, key_word_list: list, func: str
    ) -> pd.DataFrame:
        """all_trickers [summary]

        [extended_summary]

        Args:
            tickers (yf.Tickers): [description]
            key_word_list (list): [description]
            func (str): [description]

        Returns:
            pd.DataFrame: [description]
        """
        return {
            key_word: getattr(getattr(tickers.tickers, key_word), func)
            for key_word in key_word_list
        }

    def get_chart_history(
        self,
        period: str,
        interval: str,
        start: str,
        end: str,
        prepost: bool,
        actions: bool,
        auto_adjust: bool,
        proxy: str,
        threads: bool,
        group_by: str,
        progress: bool,
        **kwargs,
    ) -> None:
        """get_chart_history [summary]

        [extended_summary]

        Args:
            period (str): [description]
            interval (str): [description]
            start (str): [description]
            end (str): [description]
            prepost (bool): [description]
            actions (bool): [description]
            auto_adjust (bool): [description]
            proxy (str): [description]
            threads (bool): [description]
            group_by (str): [description]
            progress (bool): [description]
        """
        self.df = self.tickers.history(
            period=period,
            interval=interval,
            start=start,
            end=end,
            prepost=prepost,
            actions=actions,
            auto_adjust=auto_adjust,
            proxy=proxy,
            threads=threads,
            group_by=group_by,
            progress=progress,
            **kwargs,
        )

    def get_isin_code(self) -> None:
        """get_isin_code [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers, key_word_list=self.key_word_list, func="isin"
        )

    def get_major_holders(self) -> None:
        """get_major_holders [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers, key_word_list=self.key_word_list, func="major_holders"
        )

    def get_institutional_holders(self) -> None:
        """get_institutional_holders [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers,
            key_word_list=self.key_word_list,
            func="institutional_holders",
        )

    def get_mutualfund_holders(self) -> None:
        """get_mutualfund_holders [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers,
            key_word_list=self.key_word_list,
            func="mutualfund_holders",
        )

    def get_dividends(self) -> None:
        """get_dividends [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers, key_word_list=self.key_word_list, func="dividends"
        )

    def get_splits(self) -> None:
        """get_splits [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers, key_word_list=self.key_word_list, func="splits"
        )

    def get_actions(self) -> None:
        """get_actions [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers, key_word_list=self.key_word_list, func="actions"
        )

    def get_info(self) -> None:
        """get_info [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers, key_word_list=self.key_word_list, func="info"
        )

    def get_calendar(self) -> None:
        """get_calendar [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers, key_word_list=self.key_word_list, func="calendar"
        )

    def get_recommendations(self) -> None:
        """get_recommendations [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers,
            key_word_list=self.key_word_list,
            func="recommendations",
        )

    def get_earnings(self) -> None:
        """get_earnings [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers,
            key_word_list=self.key_word_list,
            func="earnings",
        )

    def get_quarterly_earnings(self) -> None:
        """get_quarterly_earnings [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers,
            key_word_list=self.key_word_list,
            func="quarterly_earnings",
        )

    def get_financials(self) -> None:
        """get_financials [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers,
            key_word_list=self.key_word_list,
            func="financials",
        )

    def get_quarterly_financials(self) -> None:
        """get_quarterly_financials [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers,
            key_word_list=self.key_word_list,
            func="quarterly_financials",
        )

    def get_balancesheet(self) -> None:
        """get_balancesheet [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers,
            key_word_list=self.key_word_list,
            func="balancesheet",
        )

    def get_quarterly_balancesheet(self) -> None:
        """get_quarterly_balancesheet [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers,
            key_word_list=self.key_word_list,
            func="quarterly_balancesheet",
        )

    def get_cashflow(self) -> None:
        """get_cashflow [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers,
            key_word_list=self.key_word_list,
            func="cashflow",
        )

    def get_quarterly_cashflow(self) -> None:
        """get_quarterly_cashflow [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers,
            key_word_list=self.key_word_list,
            func="quarterly_cashflow",
        )

    def get_sustainability(self) -> None:
        """get_sustainability [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers,
            key_word_list=self.key_word_list,
            func="sustainability",
        )

    def get_options(self) -> None:
        """get_options [summary]

        [extended_summary]
        """
        self.dict = self.all_trickers(
            tickers=self.tickers,
            key_word_list=self.key_word_list,
            func="options",
        )


class FinanceCollector:
    """[summary]

    [extended_summary]
    """

    def __init__(self) -> None:
        """__init__ [summary]

        [extended_summary]
        """
        self._builder = None

    @property
    def builder(self) -> BuilderFinanceCollector:
        """builder [summary]

        [extended_summary]

        Returns:
            BuilderFinanceCollector: [description]
        """
        return self._builder

    @builder.setter
    def builder(self, builder: BuilderFinanceCollector) -> property:
        """builder [summary]

        [extended_summary]

        Args:
            builder (BuilderFinanceCollector): [description]

        Returns:
            property: [description]
        """
        self._builder = builder

    def find_chart_histogram(
        self,
        period: str = "1mo",
        interval: str = "1d",
        start: str = None,
        end: str = None,
        prepost: bool = False,
        actions: bool = True,
        auto_adjust: bool = True,
        proxy: str = None,
        threads: bool = True,
        group_by: str = "column",
        progress: bool = True,
        **kwargs,
    ) -> None:
        """find_chart_histogram [summary]

        [extended_summary]

        Args:
            period (str, optional): [description]. Defaults to "1mo".
            interval (str, optional): [description]. Defaults to "1d".
            start (str, optional): [description]. Defaults to None.
            end (str, optional): [description]. Defaults to None.
            prepost (bool, optional): [description]. Defaults to False.
            actions (bool, optional): [description]. Defaults to True.
            auto_adjust (bool, optional): [description]. Defaults to True.
            proxy (str, optional): [description]. Defaults to None.
            threads (bool, optional): [description]. Defaults to True.
            group_by (str, optional): [description]. Defaults to "column".
            progress (bool, optional): [description]. Defaults to True.
        """
        self.builder.get_chart_history(
            period=period,
            interval=interval,
            start=start,
            end=end,
            prepost=prepost,
            actions=actions,
            auto_adjust=auto_adjust,
            proxy=proxy,
            threads=threads,
            group_by=group_by,
            progress=progress,
            **kwargs,
        )

    def find_isin_code(self) -> None:
        """find_isin_code [summary]

        [extended_summary]
        """
        self.builder.get_isin_code()

    def find_major_holders(self) -> None:
        """find_major_holders [summary]

        [extended_summary]
        """
        self.builder.get_major_holders()

    def find_institutional_holders(self) -> None:
        """find_institutional_holders [summary]

        [extended_summary]
        """
        self.builder.get_institutional_holders()

    def find_mutualfund_holders(self) -> None:
        """find_mutualfund_holders [summary]

        [extended_summary]
        """
        self.builder.get_mutualfund_holders()

    def find_dividends(self) -> None:
        """find_dividends [summary]

        [extended_summary]
        """
        self.builder.get_dividends()

    def find_splits(self) -> None:
        """find_splits [summary]

        [extended_summary]
        """
        self.builder.get_splits()

    def find_actions(self) -> None:
        """find_actions [summary]

        [extended_summary]
        """
        self.builder.get_actions()

    def find_info(self) -> None:
        """find_info [summary]

        [extended_summary]
        """
        self.builder.get_info()

    def find_calendar(self) -> None:
        """find_calendar [summary]

        [extended_summary]
        """
        self.builder.get_calendar()

    def find_recommendations(self) -> None:
        """find_recommendations [summary]

        [extended_summary]
        """
        self.builder.get_recommendations()

    def find_earnings(self) -> None:
        """find_earnings [summary]

        [extended_summary]
        """
        self.builder.get_earnings()

    def find_quarterly_earnings(self) -> None:
        """find_quarterly_earnings [summary]

        [extended_summary]
        """
        self.builder.get_quarterly_earnings()

    def find_financials(self) -> None:
        """find_financials [summary]

        [extended_summary]
        """
        self.builder.get_financials()

    def find_quarterly_financials(self) -> None:
        """find_quarterly_financials [summary]

        [extended_summary]
        """
        self.builder.get_quarterly_financials()

    def find_balancesheet(self) -> None:
        """find_balancesheet [summary]

        [extended_summary]
        """
        self.builder.get_balancesheet()

    def find_quarterly_balancesheet(self) -> None:
        """find_quarterly_balancesheet [summary]

        [extended_summary]
        """
        self.builder.get_quarterly_balancesheet()

    def find_cashflow(self) -> None:
        """find_cashflow [summary]

        [extended_summary]
        """
        self.builder.get_cashflow()

    def find_quarterly_cashflow(self) -> None:
        """find_quarterly_cashflow [summary]

        [extended_summary]
        """
        self.builder.get_quarterly_cashflow()

    def find_sustainability(self) -> None:
        """find_sustainability [summary]

        [extended_summary]
        """
        self.builder.get_sustainability()

    def find_options(self) -> None:
        """find_options [summary]

        [extended_summary]
        """
        self.builder.get_options()
