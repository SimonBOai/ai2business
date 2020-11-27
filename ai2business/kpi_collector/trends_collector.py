from __future__ import annotations

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
    def interest_over_time(self) -> None:
        pass

    @abstractmethod
    def interest_by_region(self) -> None:
        pass

    @abstractmethod
    def trending_searches(self) -> None:
        pass

    @abstractmethod
    def today_searches(self) -> None:
        pass

    @abstractmethod
    def top_charts(self) -> None:
        pass

    @abstractmethod
    def related_topics(self) -> None:
        pass

    @abstractmethod
    def related_queries(self) -> None:
        pass

    @abstractmethod
    def suggestions(self) -> None:
        pass

    @abstractmethod
    def categories(self) -> None:
        pass

    # @abstractmethod
    # def get_historical_interest(self):
    #    pass


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
        **kwargs
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
            **kwargs
        )
        self.df = pd.DataFrame()
        self.dict = dict()

    def interest_over_time(self) -> None:
        self.df = self.pytrends.interest_over_time()

    def interest_by_region(self, resolution: str, **kwargs) -> None:
        self.df = self.pytrends.interest_by_region(resolution=resolution, **kwargs)

    def trending_searches(self, trend_country: str = "united_states") -> None:
        self.df = self.pytrends.trending_searches(pn=trend_country)

    def today_searches(self, today_country: str) -> None:
        self.df = self.pytrends.today_searches(pn=today_country)

    def top_charts(self, date: int, top_country: str) -> None:
        self.df = self.pytrends.top_charts(
            date, hl=self.language, tz=self.timezone, geo=top_country
        )

    def related_topics(self) -> None:
        self.dict = self.pytrends.related_topics()

    def related_queries(self) -> None:
        self.dict = self.pytrends.related_queries()

    def suggestions(self) -> None:
        for keyword in self.key_word_list:
            self.dict[keyword] = self.pytrends.suggestions(keyword=keyword)

    def categories(self) -> None:
        self.dict = self.pytrends.categories()

    @property
    def return_dataframe(self) -> pd.DataFrame:
        return self.df

    @property
    def return_dict(self) -> dict:
        return self.dict


class TrendsCollector:
    def __init__(self) -> None:
        self._builder = None

    @property
    def builder(self) -> BuilderTrendsCollector:
        return self._builder

    @builder.setter
    def builder(self, builder: BuilderTrendsCollector) -> None:
        self._builder = builder

    def find_interest_over_time(self) -> None:
        self.builder.interest_over_time()

    def find_interest_by_region(self, resolution: str = "COUNTRY", **kwargs) -> None:
        self.builder.interest_by_region(resolution=resolution, **kwargs)

    def find_trending_searches(self, trend_country: str = "united_states") -> None:
        self.builder.trending_searches(trend_country=trend_country)

    def find_today_searches(self, today_country: str = "US") -> None:
        self.builder.today_searches(today_country=today_country)

    def find_top_charts(self, date: int, top_country: str = "GLOBAL") -> None:
        self.builder.top_charts(date=date, top_country=top_country)

    def find_related_topics(self) -> None:
        self.builder.related_topics()

    def find_related_queries(self) -> None:
        self.builder.related_queries()

    def find_suggestions(self) -> None:
        self.builder.suggestions()

    def find_categories(self) -> None:
        self.builder.categories()

    # def find_interest_over_time(self) -> None:
    #    self.builder.interest_over_time()


if __name__ == "__main__":

    trends = TrendsCollector()
    builder = DesignerTrendsCollector(["Pizza", "Baggel"])
    trends.builder = builder
    trends.find_trending_searches()
    print(builder.return_dataframe.head())
    trends.find_today_searches()
    print(builder.return_dataframe.head())
    trends.find_top_charts(2018)
    print(builder.return_dataframe.head())
    # print(type(builder.return_dataframe) == type(pd.DataFrame()))
