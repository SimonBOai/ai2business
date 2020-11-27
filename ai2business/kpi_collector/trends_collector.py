from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from pytrends.request import TrendReq
import pandas as pd


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
    def top_chart(self) -> None:
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
        key_word_list: list =["pizza", "bagel"],
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

    def interest_by_region(self, resolution: str = "COUNTRY", **kwargs) -> None:
        self.df = self.pytrends.interest_by_region(resolution=resolution, **kwargs)

    def trending_searches(self) -> None:
        self.df = self.pytrends.trending_searches(pn=self.country)

    def today_searches(self) -> None:
        self.df = self.pytrends.today_searches(pn=self.country)

    def top_chart(self, date: str) -> None:
        self.df = self.pytrends.top_chart(
            date, hl=self.language, tz=self.timezone, geo=self.country
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


if __name__ == "__main__":

    trends = TrendsCollector()
    builder = DesignerTrendsCollector()
    trends.builder = builder
    trends.find_interest_over_time()
    print(type(builder.return_dataframe) == type(pd.DataFrame()))
    