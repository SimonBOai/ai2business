""" [summary]

    [extended_summary]
"""
from abc import ABC, abstractmethod, abstractproperty

import pandas as pd
from pytrends.request import TrendReq


class BuilderTrendsCollector(ABC):
    """BuilderTrendsCollector contains the abstract properties and methods.

    `BuilderTrendsCollector` specifies the properties and methods for creating the
    different parts of the `DesignerTrendsCollector` objects.

    Args:
        ABC (class): Helper class that provides a standard way to create an ABC using
        inheritance.
    """

    @abstractproperty
    def return_dataframe(self) -> None:
        """Abstract property of return_dataframe."""
        pass

    @abstractproperty
    def return_dict(self) -> None:
        """Abstract property of return_dict."""
        pass

    @abstractmethod
    def get_interest_over_time(self) -> None:
        """Abstract method of get_interest_over_time."""
        pass

    @abstractmethod
    def get_interest_by_region(self) -> None:
        """Abstract method of get_interest_by_region."""
        pass

    @abstractmethod
    def get_trending_searches(self) -> None:
        """Abstract method of get_trending_searches."""
        pass

    @abstractmethod
    def get_today_searches(self) -> None:
        """Abstract method of get_today_searches."""
        pass

    @abstractmethod
    def get_top_charts(self) -> None:
        """Abstract method of get_top_charts."""
        pass

    @abstractmethod
    def get_related_topics(self) -> None:
        """Abstract method of get_related_topics."""
        pass

    @abstractmethod
    def get_related_queries(self) -> None:
        """Abstract extended_summary of get_related_queries."""
        pass

    @abstractmethod
    def get_suggestions(self) -> None:
        """Abstract method of get_suggestions."""
        pass

    @abstractmethod
    def get_categories(self) -> None:
        """Abstract method of get_categories."""
        pass

    @abstractmethod
    def get_historical_interest(self):
        """Abstract method of get_historical_interest."""
        pass


class DesignerTrendsCollector(BuilderTrendsCollector):
    """DesignerTrendsCollector contains the specific implementation of
    `BuilderTrendsCollector`.

    `DesignerTrendsCollector` contains the specific implementation of
    `BuilderTrendsCollector` based on the external library `pytrends`.

    Args:
        BuilderTrendsCollector (class): Abstract class that provides the implementations
        of the properties and methods.
    """

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
        """Initialization of DesignerTrendsCollector

        Args:
            key_word_list (list): Keyword-list with the items to search for.
            timeframe (str, optional): Time frame, respectively, period to search for.
            Defaults to "today 5-y".
            language (str, optional): Search language. Defaults to "en-US".
            category (int, optional): Define a specific [search category](https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories).
            Defaults to 0.
            timezone (int, optional): [Search timezone](https://developers.google.com/maps/documentation/timezone/overview).
            Defaults to 360.
            country (str, optional): The country, where to search for. Defaults to "".
            property_filter (str, optional): Property filer of the search; only in news,
            images, YouTube, shopping. Defaults to "".
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
        """Return the search results as dataframe.

        Returns:
            pd.DataFrame: Two-dimensional, size-mutable, homogenous tabular data,
            which contains the trend-search results as time-series.
        """
        return self.df

    @property
    def return_dict(self) -> dict:
        """Return the search results as dictionary.

        Returns:
            dict: Multi-dimensional, size-mutable, mainly heterogeneous data as
            dictionary, which contains the `clustered` or `nested` trend-search results.
        """
        return self.dict

    def get_interest_over_time(self) -> None:
        """Request data from a interest over time search."""
        self.df = self.pytrends.interest_over_time()

    def get_interest_by_region(self, resolution: str, **kwargs) -> None:
        """Request data from a interest by region search.

        Args:
            resolution (str): The resolution of the subregion.
        """
        self.df = self.pytrends.interest_by_region(resolution=resolution, **kwargs)

    def get_trending_searches(self, trend_country: str = "united_states") -> None:
        """Request data from a search by country.

        Args:
            trend_country (str, optional): Name of the country of intrest. Defaults to
            "united_states".
        """
        self.df = self.pytrends.trending_searches(pn=trend_country)

    def get_today_searches(self, today_country: str) -> None:
        """Request data from the daily search trends.

        Args:
            today_country (str): Name of the country of intrest.
        """
        self.df = self.pytrends.today_searches(pn=today_country)

    def get_top_charts(self, date: int, top_country: str) -> None:
        """Request data from a top charts search.

        Args:
            date (int): Year
            top_country (str): Name of the country of intrest.
        """
        self.df = self.pytrends.top_charts(
            date, hl=self.language, tz=self.timezone, geo=top_country
        )

    def get_related_topics(self) -> None:
        """Request data of a related topics based on the keyword."""
        self.dict = self.pytrends.related_topics()

    def get_related_queries(self) -> None:
        """Request data of a related queries based on the keyword."""
        self.dict = self.pytrends.related_queries()

    def get_suggestions(self) -> None:
        """Request data from keyword suggestion dropdown search."""
        for keyword in self.key_word_list:
            self.dict[keyword] = self.pytrends.suggestions(keyword=keyword)

    def get_categories(self) -> None:
        """Request available categories data for the current search."""
        self.dict = self.pytrends.categories()

    def get_historical_interest(
        self,
        year_start: int,
        month_start: int,
        day_start: int,
        hour_start: int,
        year_end: int,
        month_end: int,
        day_end: int,
        hour_end: int,
        **kwargs,
    ) -> None:
        """Request data from a hour-grided time search.

        Args:
            year_start (int): Starting year
            month_start (int): Starting month
            day_start (int): Starting day
            hour_start (int): Starting hour
            year_end (int): Final year
            month_end (int): Final month
            day_end (int): Final day
            hour_end (int): Final hour
        """
        self.df = self.pytrends.get_historical_interest(
            keywords=self.key_word_list,
            year_start=year_start,
            month_start=month_start,
            day_start=day_start,
            hour_start=hour_start,
            year_end=year_end,
            month_end=month_end,
            day_end=day_end,
            hour_end=hour_end,
            cat=self.category,
            geo=self.country,
            gprop=self.property_filter,
            **kwargs,
        )


class TrendsCollector:
    """TrendsCollector is in charge for executing the functions.

    During the execution, `TrendsCollector` can construct several product variations
    using the same building steps.
    """

    def __init__(self) -> None:
        """Initialize a fresh and empty builder."""
        self._builder = None

    @property
    def builder(self) -> BuilderTrendsCollector:
        """Builder as a property with value None.

        Returns:
            BuilderTrendsCollector: A builder class, that contains the abstract
            properties and methods.
        """
        return self._builder

    @builder.setter
    def builder(self, builder: BuilderTrendsCollector) -> property:
        """Sets the builder according to BuilderTrendsCollector.

        Args:
            builder (BuilderTrendsCollector): A builder class, that contains the
            abstract properties and methods.

        Returns:
            property: A method or property of `BuilderTrendsCollector`.
        """
        self._builder = builder

    def find_interest_over_time(self) -> None:
        """Perform a interest over time search."""
        self.builder.get_interest_over_time()

    def find_interest_by_region(self, resolution: str = "COUNTRY", **kwargs) -> None:
        """Perform a interest over region search.

        Args:
            resolution (str, optional): The resolution of the subregion. Defaults to "COUNTRY".
        """
        self.builder.get_interest_by_region(resolution=resolution, **kwargs)

    def find_trending_searches(self, trend_country: str = "united_states") -> None:
        """Performa a search trend analysis.

        Args:
            trend_country (str, optional): Name of the country of intrest. Defaults to
            "united_states".
        """
        self.builder.get_trending_searches(trend_country=trend_country)

    def find_today_searches(self, today_country: str = "US") -> None:
        """Perform a search analysis about today's hot topics.

        Args:
            today_country (str, optional): Name of the country of intrest. Defaults to
            "US".
        """
        self.builder.get_today_searches(today_country=today_country)

    def find_top_charts(self, date: int, top_country: str = "GLOBAL") -> None:
        """Perform a search chart analysis.

        Args:
            date (int): [description]
            top_country (str, optional): [description]. Defaults to "GLOBAL".
        """
        try:
            self.builder.get_top_charts(date=date, top_country=top_country)
        except IndexError as exc:
            print(f"ERROR: {exc} -> Date is illegal!")

    def find_related_topics(self) -> None:
        """Find the related topics to a keyword."""
        self.builder.get_related_topics()

    def find_related_queries(self) -> None:
        """Find the related queries to a keyword."""
        self.builder.get_related_queries()

    def find_suggestions(self) -> None:
        """Find suggestions for a given keyword."""
        self.builder.get_suggestions()

    def find_categories(self) -> None:
        """Find the current search categories."""
        self.builder.get_categories()

    def find_historical_interest(
        self,
        year_start: int,
        month_start: int,
        day_start: int,
        hour_start: int,
        year_end: int,
        month_end: int,
        day_end: int,
        hour_end: int,
        **kwargs,
    ) -> None:
        """Preform a search trend analysis for a given time period in hour steps.

        Args:
            year_start (int): Starting year
            month_start (int): Starting month
            day_start (int): Starting day
            hour_start (int): Starting hour
            year_end (int): Final year
            month_end (int): Final month
            day_end (int): Final day
            hour_end (int): Final hour
        """
        try:
            self.builder.get_historical_interest(
                year_start=year_start,
                month_start=month_start,
                day_start=day_start,
                hour_start=hour_start,
                year_end=year_end,
                month_end=month_end,
                day_end=day_end,
                hour_end=hour_end,
                **kwargs,
            )
        except ValueError as exc:
            print(f"ERROR: {exc} -> Date is illegal!")
