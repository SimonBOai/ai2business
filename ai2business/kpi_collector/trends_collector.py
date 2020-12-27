"""Trends Collection Module: Collecting Search Trends via http-API."""
from abc import ABC, abstractmethod, abstractproperty
from typing import Callable

import pandas as pd
from pytrends.request import TrendReq


class BuilderTrendsCollector(ABC):
    """BuilderTrendsCollector contains the abstract properties and methods.

    `BuilderTrendsCollector` specifies the properties and methods for creating the
    different parts of the `DesignerTrendsCollector` objects.

    Args:
        ABC (class): Helper class that provides a standard way to create an ABC using inheritance.
    """

    @abstractmethod
    def reset(self) -> None:
        """Abstract method of reset."""

    @abstractproperty
    def trends(self) -> None:
        """Abstract property of trends."""

    @abstractmethod
    def get_interest_over_time(self) -> None:
        """Abstract method of get_interest_over_time."""

    @abstractmethod
    def get_interest_by_region(self) -> None:
        """Abstract method of get_interest_by_region."""

    @abstractmethod
    def get_trending_searches(self) -> None:
        """Abstract method of get_trending_searches."""

    @abstractmethod
    def get_today_searches(self) -> None:
        """Abstract method of get_today_searches."""

    @abstractmethod
    def get_top_charts(self) -> None:
        """Abstract method of get_top_charts."""

    @abstractmethod
    def get_related_topics(self) -> None:
        """Abstract method of get_related_topics."""

    @abstractmethod
    def get_related_queries(self) -> None:
        """Abstract extended_summary of get_related_queries."""

    @abstractmethod
    def get_suggestions(self) -> None:
        """Abstract method of get_suggestions."""

    @abstractmethod
    def get_categories(self) -> None:
        """Abstract method of get_categories."""

    @abstractmethod
    def get_historical_interest(self):
        """Abstract method of get_historical_interest."""


class TrendProduct:
    """TrendProduct contains the dictionary and the return value of it."""

    def __init__(self) -> None:
        """Initialization of TrendProduct"""
        self.product_parts = {}

    def add_product(self, key: Callable, value: pd.DataFrame or dict) -> None:
        """Add the components of the trend search to the dictionary.

        Args:
            key (Callable): Used trend search function
            value (pd.DataFrame or dict): Return value as dataframe or dictionary of the function.
        """
        self.product_parts[key.__name__] = value

    @property
    def list_product_parts(self) -> str:
        """List of the product parts in the dictionary."""
        return f"Product parts: {', '.join(self.product_parts)}"

    @property
    def return_product(self) -> dict:
        """Returns the product as a dictionary

        Returns:
            dict: The product dictionary contains the product and ist function name as `key`.
        """
        return self.product_parts


class DesignerTrendsCollector(BuilderTrendsCollector):
    """DesignerTrendsCollector contains the specific implementation of
    `BuilderTrendsCollector`.

    `DesignerTrendsCollector` contains the specific implementation of
    `BuilderTrendsCollector` based on the external library `pytrends`.

    Args:
        BuilderTrendsCollector (class): Abstract class that provides the implementations of the properties and methods.
    """

    def __init__(
        self,
        keyword_list: list,
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
            keyword_list (list): Keyword-list with the items to search for.
            timeframe (str, optional): Time frame, respectively, period to search for.
            Defaults to "today 5-y".
            language (str, optional): Search language. Defaults to "en-US".
            category (int, optional): Define a specific [search category](https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories). Defaults to 0.
            timezone (int, optional): [Search timezone](https://developers.google.com/maps/documentation/timezone/overview). Defaults to 360.
            country (str, optional): The country, where to search for. Defaults to "".
            property_filter (str, optional): Property filer of the search; only in news, images, YouTube, shopping. Defaults to "".
        """
        self.keyword_list = keyword_list
        self.timeframe = timeframe
        self.language = language
        self.category = category
        self.timezone = timezone
        self.country = country
        self.property_filter = property_filter

        self.pytrends = TrendReq(hl=self.language, tz=self.timezone, **kwargs)
        self.pytrends.build_payload(
            kw_list=self.keyword_list,
            cat=self.category,
            timeframe=self.timeframe,
            geo=self.country,
            gprop=self.property_filter,
        )
        self.reset()

    def reset(self) -> None:
        """Reset the product to empty."""
        self._product = TrendProduct()

    @property
    def trends(self) -> TrendProduct:
        """Return the trend results.

        Returns:
            TrendProduct: (class) TrendProduct contains the dictionary and the return value of it.
        """
        product = self._product
        self.reset()
        return product

    def get_interest_over_time(self) -> None:
        """Request data from a interest over time search."""
        self._product.add_product(
            key=self.pytrends.interest_over_time,
            value=self.pytrends.interest_over_time(),
        )

    def get_interest_by_region(self, resolution: str, **kwargs) -> None:
        """Request data from a interest by region search.

        Args:
            resolution (str): The resolution of the subregion.
        """
        self._product.add_product(
            key=self.pytrends.interest_by_region,
            value=self.pytrends.interest_by_region(resolution=resolution, **kwargs),
        )

    def get_trending_searches(self, trend_country: str) -> None:
        """Request data from a search by country.

        Args:
            trend_country (str, optional): Name of the country of intrest. Defaults to "united_states".
        """
        self._product.add_product(
            key=self.pytrends.trending_searches,
            value=self.pytrends.trending_searches(pn=trend_country),
        )

    def get_today_searches(self, today_country: str) -> None:
        """Request data from the daily search trends.

        Args:
            today_country (str): Name of the country of intrest.
        """
        self._product.add_product(
            key=self.pytrends.today_searches,
            value=self.pytrends.today_searches(pn=today_country),
        )

    def get_top_charts(self, date: int, top_country: str) -> None:
        """Request data from a top charts search.

        Args:
            date (int): Year
            top_country (str): Name of the country of intrest.
        """
        self._product.add_product(
            key=self.pytrends.top_charts,
            value=self.pytrends.top_charts(
                date, hl=self.language, tz=self.timezone, geo=top_country
            ),
        )

    def get_related_topics(self) -> None:
        """Request data of a related topics based on the keyword."""
        self._product.add_product(
            key=self.pytrends.related_topics, value=self.pytrends.related_topics()
        )

    def get_related_queries(self) -> None:
        """Request data of a related queries based on the keyword."""
        self._product.add_product(
            key=self.pytrends.related_queries,
            value=self.pytrends.related_queries(),
        )

    def get_suggestions(self) -> None:
        """Request data from keyword suggestion dropdown search."""
        self._product.add_product(
            key=self.pytrends.suggestions,
            value={
                keyword: self.pytrends.suggestions(keyword=keyword)
                for keyword in self.keyword_list
            },
        )

    def get_categories(self) -> None:
        """Request available categories data for the current search."""
        self._product.add_product(
            key=self.pytrends.categories,
            value=self.pytrends.categories(),
        )

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
        self._product.add_product(
            key=self.pytrends.get_historical_interest,
            value=self.pytrends.get_historical_interest(
                keywords=self.keyword_list,
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
            ),
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
            BuilderTrendsCollector: A builder class, that contains the abstract properties and methods.
        """
        return self._builder

    @builder.setter
    def builder(self, builder: BuilderTrendsCollector) -> property:
        """Sets the builder according to BuilderTrendsCollector.

        Args:
            builder (BuilderTrendsCollector): A builder class, that contains the abstract properties and methods.

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
            trend_country (str, optional): Name of the country of intrest. Defaults to "united_states".
        """
        self.builder.get_trending_searches(trend_country=trend_country)

    def find_today_searches(self, today_country: str = "US") -> None:
        """Perform a search analysis about today's hot topics.

        Args:
            today_country (str, optional): Name of the country of intrest. Defaults to "US".
        """
        self.builder.get_today_searches(today_country=today_country)

    def find_top_charts(self, date: int, top_country: str = "GLOBAL") -> None:
        """Perform a search chart analysis.

        Args:
            date (int): Year
            top_country (str, optional): [description]. Defaults to "GLOBAL".
        """
        try:
            self.builder.get_top_charts(date=date, top_country=top_country)
        except IndexError as exc:
            print(f"ERROR: {exc} -> Date is illegal!")

    def find_related_topics(self) -> None:
        """Perform a search about the related topics to a keyword."""
        self.builder.get_related_topics()

    def find_related_queries(self) -> None:
        """Perform a search about the related queries to a keyword."""
        self.builder.get_related_queries()

    def find_suggestions(self) -> None:
        """Perform a search about suggestions for a given keyword."""
        self.builder.get_suggestions()

    def find_categories(self) -> None:
        """Perform a search about the current search categories."""
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

    def make_wordcloud(self) -> None:
        """Make a worldcloud of related words and their suggestions.

        Note:
        ---

        The summation of different search engines can cause timeout errors due to lenght of the search.
        """
        self.builder.get_related_topics()
        self.builder.get_related_queries()
        self.builder.get_suggestions()
        self.builder.get_categories()
