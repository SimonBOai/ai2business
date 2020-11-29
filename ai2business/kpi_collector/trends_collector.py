# from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty

import pandas as pd
from pytrends.request import TrendReq


class BuilderTrendsCollector(ABC):
    @abstractproperty
    def return_dataframe(self) -> None:
        pass

    @abstractproperty
    def return_dict(self) -> None:
        pass

    @abstractmethod
    def get_interest_over_time(self) -> None:
        pass

    @abstractmethod
    def get_interest_by_region(self) -> None:
        pass

    @abstractmethod
    def get_trending_searches(self) -> None:
        pass

    @abstractmethod
    def get_today_searches(self) -> None:
        pass

    @abstractmethod
    def get_top_charts(self) -> None:
        pass

    @abstractmethod
    def get_related_topics(self) -> None:
        pass

    @abstractmethod
    def get_related_queries(self) -> None:
        pass

    @abstractmethod
    def get_suggestions(self) -> None:
        pass

    @abstractmethod
    def get_categories(self) -> None:
        pass

    @abstractmethod
    def get_historical_interest(self):
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
        self.dict = dict()

    @property
    def return_dataframe(self) -> pd.DataFrame:
        return self.df

    @property
    def return_dict(self) -> dict:
        return self.dict

    def get_interest_over_time(self) -> None:
        self.df = self.pytrends.interest_over_time()

    def get_interest_by_region(self, resolution: str, **kwargs) -> None:
        self.df = self.pytrends.interest_by_region(resolution=resolution, **kwargs)

    def get_trending_searches(self, trend_country: str = "united_states") -> None:
        self.df = self.pytrends.trending_searches(pn=trend_country)

    def get_today_searches(self, today_country: str) -> None:
        self.df = self.pytrends.today_searches(pn=today_country)

    def get_top_charts(self, date: int, top_country: str) -> None:
        self.df = self.pytrends.top_charts(
            date, hl=self.language, tz=self.timezone, geo=top_country
        )

    def get_related_topics(self) -> None:
        self.dict = self.pytrends.related_topics()

    def get_related_queries(self) -> None:
        self.dict = self.pytrends.related_queries()

    def get_suggestions(self) -> None:
        for keyword in self.key_word_list:
            self.dict[keyword] = self.pytrends.suggestions(keyword=keyword)

    def get_categories(self) -> None:
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
    def __init__(self) -> None:
        self._builder = None

    @property
    def builder(self) -> BuilderTrendsCollector:
        return self._builder

    @builder.setter
    def builder(self, builder: BuilderTrendsCollector) -> property:
        self._builder = builder

    def find_interest_over_time(self) -> None:
        self.builder.get_interest_over_time()

    def find_interest_by_region(self, resolution: str = "COUNTRY", **kwargs) -> None:
        self.builder.get_interest_by_region(resolution=resolution, **kwargs)

    def find_trending_searches(self, trend_country: str = "united_states") -> None:
        self.builder.get_trending_searches(trend_country=trend_country)

    def find_today_searches(self, today_country: str = "US") -> None:
        self.builder.get_today_searches(today_country=today_country)

    def find_top_charts(self, date: int, top_country: str = "GLOBAL") -> None:
        try:
            self.builder.get_top_charts(date=date, top_country=top_country)
        except IndexError as exc:
            print(f"ERROR: {exc} -> Date is illegal!")
            pass

    def find_related_topics(self) -> None:
        self.builder.get_related_topics()

    def find_related_queries(self) -> None:
        self.builder.get_related_queries()

    def find_suggestions(self) -> None:
        self.builder.get_suggestions()

    def find_categories(self) -> None:
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
            pass
