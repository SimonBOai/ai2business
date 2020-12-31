"""Data Visualization Module: Visualization of data and its first principal properties."""
from abc import ABC, abstractmethod, abstractproperty
from typing import Callable

import matplotlib.pyplot as plt
import missingno as mss
import pandas as pd
import seaborn as sns


class BuilderDataVisualization(ABC):
    """BuilderDataVisualization contains the abstract properties and methods.

    `BuilderDataVisualization` specifies the properties and methods for creating the
    different parts of the `DesignerDataVisualization` objects.

    Args:
        ABC (class): Helper class that provides a standard way to create an ABC using inheritance.
    """

    @abstractmethod
    def reset(self) -> None:
        """Abstract method of reset."""

    @abstractproperty
    def data_figure(self) -> None:
        """Abstract property of data_figure."""

    @abstractmethod
    def get_nullity_matrix(self) -> None:
        """Abstract method of get_nullity_matrix."""

    @abstractmethod
    def get_nullity_bar(self) -> None:
        """Abstract method of get_nullity_bar."""

    @abstractmethod
    def get_nullity_heatmap(self) -> None:
        """Abstract method of get_nullity_heatmap."""

    @abstractmethod
    def get_nullity_dendrogram(self) -> None:
        """Abstract method of get_nullity_dendrogram."""


class DataVisualizationProduct:
    """DataVisualizationProduct contains the dictionary and the return value of it."""

    def __init__(self) -> None:
        """Initialization of DataVisualizationProduct."""
        self.product_parts = {}

    def add_product(self, key: Callable, value: plt.subplot) -> None:
        """Add the components of the data visualization to the dictionary.

        Args:
            key (Callable): Used data visualization search function
            value (plt.subplot): Return value as `matplotlib`-class of subplots.
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


class DesignerDataVisualization(BuilderDataVisualization):
    def __init__(
        self, df: pd.DataFrame, fontsize: int = 16, filter: str = None
    ) -> None:

        self.df = df
        self.filter = filter
        self.reset()

    def reset(self) -> None:
        """Reset the product to empty."""
        self._product = DataVisualizationProduct()

    @property
    def data_figure(self) -> DataVisualizationProduct:
        """Return the results of the figure class of the data visualization.

        Returns:
            DataVisualizationProduct (class): DataVisualizationProduct contains the dictionary and the return value of it.
        """
        product = self._product
        self.reset()
        return product

    def get_nullity_matrix(self) -> None:
        """Generates the nullity matrix."""
        self._product.add_product(
            key=self.get_nullity_matrix,
            value=mss.matrix(self.df),
        )

    def get_nullity_bar(self) -> None:
        """Generates the nullity bar."""
        self._product.add_product(
            key=self.get_nullity_bar,
            value=mss.bar(self.df),
        )

    def get_nullity_heatmap(self) -> None:
        """Generates the nullity heatmap."""
        self._product.add_product(
            key=self.get_nullity_heatmap,
            value=mss.heatmap(self.df, cmap="seismic"),
        )

    def get_nullity_dendrogram(self) -> None:
        """ Generates the nullity dendrogram."""
        self._product.add_product(
            key=self.get_nullity_dendrogram,
            value=mss.dendrogram(self.df),
        )


class DataVisualization:
    """DataVisualization is in charge of executing the functions.

    During the execution, `DataVisualization` can construct several product
    variations using the same building steps.
    """

    def __init__(self) -> None:
        """Initialize a fresh and empty builder."""
        self._builder = None

    @property
    def builder(self) -> BuilderDataVisualization:
        """Builder as a property with value None.

        Returns:
            BuilderDataVisualization: A builder class, that contains the abstract properties and methods.
        """
        return self._builder

    @builder.setter
    def builder(self, builder: BuilderDataVisualization) -> property:
        """Sets the builder according to BuilderDataVisualization.

        Args:
            builder (BuilderDataVisualization): A builder class, that contains the abstract properties and methods.

        Returns:
            property: A method or property of `BuilderDataVisualization`.
        """
        self._builder = builder

    def visual_missing_data(self) -> None:
        """Visualizing possible missing data.

        Possible missing data will be visualized by:

        1. nullity matrix highlights out patterns and structures in data completion.
        2. nullity bar shows the available data as a series of single bars.
        3. nullity heatmap point out the correlation between the presence and absence data.
        4. nullity dendrogram visualize the correlate variable completion.

        !!! example "Note"

            For more information see: [https://github.com/ResidentMario/missingno](https://github.com/ResidentMario/missingno)

        """
        self.builder.get_nullity_matrix()
        self.builder.get_nullity_bar()
        self.builder.get_nullity_heatmap()
        self.builder.get_nullity_dendrogram()
