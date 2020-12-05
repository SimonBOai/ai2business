""" [summary]

    [extended_summary]
"""
from abc import ABC, abstractmethod, abstractproperty

import pandas as pd
from pytrends.request import TrendReq


class BuilderTrendsCollector(ABC):
    """BuilderTrendsCollector [summary]

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

    @abstractmethod
    def get_interest_over_time(self) -> None:
        """get_interest_over_time [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_interest_by_region(self) -> None:
        """get_interest_by_region [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_trending_searches(self) -> None:
        """get_trending_searches [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_today_searches(self) -> None:
        """get_today_searches [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_top_charts(self) -> None:
        """get_top_charts [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_related_topics(self) -> None:
        """get_related_topics [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_related_queries(self) -> None:
        """get_related_queries [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_suggestions(self) -> None:
        """get_suggestions [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_categories(self) -> None:
        """get_categories [summary]

        [extended_summary]
        """
        pass

    @abstractmethod
    def get_historical_interest(self):
        """get_historical_interest [summary]

        [extended_summary]
        """
        pass


class DesignerTrendsCollector(BuilderTrendsCollector):
    def __init__(
        self,
        key_word_list: list,
        timeframe: str = "today 5-y",
        language: str = "en-US",
        category: int = 0,
        timezone: int = 360,
        country: str = "",
        property_filter="",
        **kwargs,
    ) -> None:
        """__init__ [summary]

        [extended_summary]

        Args:
            key_word_list (list): [description]
            timeframe (str, optional): [description]. Defaults to "today 5-y".
            language (str, optional): [description]. Defaults to "en-US".
            category (int, optional): [description]. Defaults to 0.
            timezone (int, optional): [description]. Defaults to 360.
            country (str, optional): [description]. Defaults to "".
            property_filter (str, optional): [description]. Defaults to "".
        """
        self.key_word_list = key_word_list
        self.timeframe = timeframe
        self.language = language
        self.category = category
        self.timezone = timezone
        self.country = country
        self.property_filter = property_filter

        self.pytrends = TrendReq(hl=self.language, tz=self.timezone, **kwargs)
        self.pytrends.build_payload(
            kw_list=self.key_word_list,
            cat=self.category,
            timeframe=self.timeframe,
            geo=self.country,
            gprop=self.property_filter,
            **kwargs,
        )
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

    def get_interest_over_time(self) -> None:
        """get_interest_over_time [summary]

        [extended_summary]
        """
        self.df = self.pytrends.interest_over_time()

    def get_interest_by_region(self, resolution: str, **kwargs) -> None:
        """get_interest_by_region [summary]

        [extended_summary]

        Args:
            resolution (str): [description]
        """
        self.df = self.pytrends.interest_by_region(resolution=resolution, **kwargs)

    def get_trending_searches(self, trend_country: str = "united_states") -> None:
        """get_trending_searches [summary]

        [extended_summary]

        Args:
            trend_country (str, optional): [description]. Defaults to "united_states".
        """
        self.df = self.pytrends.trending_searches(pn=trend_country)

    def get_today_searches(self, today_country: str) -> None:
        """get_today_searches [summary]

        [extended_summary]

        Args:
            today_country (str): [description]
        """
        self.df = self.pytrends.today_searches(pn=today_country)

    def get_top_charts(self, date: int, top_country: str) -> None:
        """get_top_charts [summary]

        [extended_summary]

        Args:
            date (int): [description]
            top_country (str): [description]
        """
        self.df = self.pytrends.top_charts(
            date, hl=self.language, tz=self.timezone, geo=top_country
        )

    def get_related_topics(self) -> None:
        """get_related_topics [summary]

        [extended_summary]
        """
        self.dict = self.pytrends.related_topics()

    def get_related_queries(self) -> None:
        """get_related_queries [summary]

        [extended_summary]
        """
        self.dict = self.pytrends.related_queries()

    def get_suggestions(self) -> None:
        """get_suggestions [summary]

        [extended_summary]
        """
        for keyword in self.key_word_list:
            self.dict[keyword] = self.pytrends.suggestions(keyword=keyword)

    def get_categories(self) -> None:
        """get_categories [summary]

        [extended_summary]
        """
        self.dict = self.pytrends.categories()

    def get_historical_interest(
        self,
        year_start,
        month_start,
        day_start,
        hour_start,
        year_end,
        day_end,
        hour_end,
        **kwargs,
    ) -> None:
        """get_historical_interest [summary]

        [extended_summary]

        Args:
            year_start ([type]): [description]
            month_start ([type]): [description]
            day_start ([type]): [description]
            hour_start ([type]): [description]
            year_end ([type]): [description]
            day_end ([type]): [description]
            hour_end ([type]): [description]
        """        
        self.df = self.pytrends.get_historical_interest(
            keywords=self.key_word_list,
            year_start=year_start,
            month_start=month_start,
            day_start=day_start,
            hour_start=hour_start,
            year_end=year_end,
            day_end=day_end,
            hour_end=hour_end,
            cat=self.category,
            geo=self.country,
            gprop=self.property_filter,
            **kwargs,
        )


class TrendsCollector:
    """ [summary]

    [extended_summary]
    """
    def __init__(self) -> None:
        """__init__ [summary]

        [extended_summary]
        """
        self._builder = None

    @property
    def builder(self) -> BuilderTrendsCollector:
        """builder [summary]

        [extended_summary]

        Returns:
            BuilderTrendsCollector: [description]
        """
        return self._builder

    @builder.setter
    def builder(self, builder: BuilderTrendsCollector) -> property:
        """builder [summary]

        [extended_summary]

        Args:
            builder (BuilderTrendsCollector): [description]

        Returns:
            property: [description]
        """
        self._builder = builder

    def find_interest_over_time(self) -> None:
        """find_interest_over_time [summary]

        [extended_summary]
        """
        self.builder.get_interest_over_time()

    def find_interest_by_region(self, resolution: str = "COUNTRY", **kwargs) -> None:
        """find_interest_by_region [summary]

        [extended_summary]

        Args:
            resolution (str, optional): [description]. Defaults to "COUNTRY".
        """
        self.builder.get_interest_by_region(resolution=resolution, **kwargs)

    def find_trending_searches(self, trend_country: str = "united_states") -> None:
        """find_trending_searches [summary]

        [extended_summary]

        Args:
            trend_country (str, optional): [description]. Defaults to "united_states".
        """
        self.builder.get_trending_searches(trend_country=trend_country)

    def find_today_searches(self, today_country: str = "US") -> None:
        """find_today_searches [summary]

        [extended_summary]

        Args:
            today_country (str, optional): [description]. Defaults to "US".
        """
        self.builder.get_today_searches(today_country=today_country)

    def find_top_charts(self, date: int, top_country: str = "GLOBAL") -> None:
        """find_top_charts [summary]

        [extended_summary]

        Args:
            date (int): [description]
            top_country (str, optional): [description]. Defaults to "GLOBAL".
        """
        try:
            self.builder.get_top_charts(date=date, top_country=top_country)
        except IndexError as exc:
            print(f"ERROR: {exc} -> Date is illegal!")

    def find_related_topics(self) -> None:
        """find_related_topics [summary]

        [extended_summary]
        """        
        self.builder.get_related_topics()

    def find_related_queries(self) -> None:
        """find_related_queries [summary]

        [extended_summary]
        """        
        self.builder.get_related_queries()

    def find_suggestions(self) -> None:
        """find_suggestions [summary]

        [extended_summary]
        """        
        self.builder.get_suggestions()

    def find_categories(self) -> None:
        """find_categories [summary]

        [extended_summary]
        """        
        self.builder.get_categories()

    def find_historical_interest(
        self,
        year_start,
        month_start,
        day_start,
        hour_start,
        year_end,
        day_end,
        hour_end,
        **kwargs,
    ) -> None:
        """find_historical_interest [summary]

        [extended_summary]

        Args:
            year_start ([type]): [description]
            month_start ([type]): [description]
            day_start ([type]): [description]
            hour_start ([type]): [description]
            year_end ([type]): [description]
            day_end ([type]): [description]
            hour_end ([type]): [description]
        """        
        try:
            self.builder.get_historical_interest(
                year_start=year_start,
                month_start=month_start,
                day_start=day_start,
                hour_start=hour_start,
                year_end=year_end,
                day_end=day_end,
                hour_end=hour_end,
                **kwargs,
            )
        except ValueError as exc:
            print(f"ERROR: {exc} -> Date is illegal!")
