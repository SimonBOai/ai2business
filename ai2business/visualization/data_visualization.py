"""Data Visualization Module: Visualization of data and its first principal properties."""
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from secrets import token_hex
from typing import Callable

import matplotlib.pyplot as plt
import missingno as mss
import numpy as np
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
    def get_lineplot(self) -> None:
        """Abstract method of get_lineplot."""

    @abstractmethod
    def get_pointplot(self) -> None:
        """Abstract method of get_pointplot."""

    @abstractmethod
    def get_scatterplot(self) -> None:
        """Abstract method of get_scatterplot."""

    @abstractmethod
    def get_swarmplot(self) -> None:
        """Abstract method of get_swarmplot."""

    @abstractmethod
    def get_distributionplot(self) -> None:
        """Abstract method of get_distributionplot."""

    @abstractmethod
    def get_relationalplot(self) -> None:
        """Abstract method of get_relationalplot."""

    @abstractmethod
    def get_categoryplot(self) -> None:
        """Abstract method of get_categoryplot."""

    @abstractmethod
    def get_boxplot(self) -> None:
        """Abstract method of get_boxplot."""

    @abstractmethod
    def get_boxenplot(self) -> None:
        """Abstract method of get_boxenplot."""

    @abstractmethod
    def get_stripplot(self) -> None:
        """Abstract method of get_stripplot."""

    @abstractmethod
    def get_hexagonplot(self) -> None:
        """Abstract method of get_hexagonplot."""

    @abstractmethod
    def get_histogramplot(self) -> None:
        """Abstract method of get_histogramplot."""

    @abstractmethod
    def get_violinplot(self) -> None:
        """Abstract method of get_violinplot."""

    @abstractmethod
    def get_residualplot(self) -> None:
        """Abstract method of get_residualplot."""

    @abstractmethod
    def get_regressionplot(self) -> None:
        """Abstract method of get_regressionplot."""

    @abstractmethod
    def get_density_mapplot(self) -> None:
        """Abstract method of get_densitymapplot."""

    @abstractmethod
    def get_kerneldensity_mapplot(self) -> None:
        """Abstract method of get_kerneldensity_mapplot."""

    @abstractmethod
    def get_cluster_mapplot(self) -> None:
        """Abstract method of get_cluster_mapplot."""

    @abstractmethod
    def get_heatmapplot(self) -> None:
        """Abstract method of get_heatmapp."""

    @abstractmethod
    def get_correlationpplot(self) -> None:
        """Abstract method of get_correlationpplot."""

    @abstractmethod
    def get_diagonal_correlationpplot(self) -> None:
        """Abstract method of get_diagonal_correlationpplot."""

    def get_regression_mapplot(self) -> None:
        """Abstract method of get_marginalplot."""

    @abstractmethod
    def get_pairmapplot(self) -> None:
        """Abstract method of get_pairmapplot."""

    @abstractmethod
    def get_complex_pairmapplot(self) -> None:
        """Abstract method of get_complex_pairmapplot."""

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

    def save_all_figures(self, folder: str = None):
        """Saving all figures of the product dictionary.

        All figures will be saved at `.` or in subfoalder as a combination of the current
        `key` and a 16 Bytes hex-token.

        Args:
            folder (str, optional): Name of the subfolder, which will be created if not exists. Defaults to None.
        """
        path = Path(".")
        for key, value in self.product_parts.items():
            if folder:
                path = Path(folder)
                path.mkdir(exist_ok=True)
            value.savefig(path.joinpath(f"{key}_{token_hex(16)}.png"))


class DesignerDataVisualization(BuilderDataVisualization):
    """DesignerDataVisualization contains the specific implementation of
    `BuilderDataVisualization`.

    `DesignerDataVisualization` contains the specific implementation of
    `BuilderDataVisualization` based on the external libraries:

    1. [missingno](https://github.com/ResidentMario/missingno)
    2. [seaborn](https://github.com/mwaskom/seaborn)

    Args:
        BuilderDataVisualization (class): Abstract class that provides the implementations of the properties and methods.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Intialization of DesignerDataVisualization.

        Args:
            df (pd.DataFrame): pandas DataFrame.
        """
        self.df = df
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

    def get_lineplot(self) -> None:
        """get_lineplot [summary]

        [extended_summary]
        """
        self._product.add_product(
            key=self.get_lineplot,
            value=sns.lineplot(data=self.df).get_figure(),
        )

    def get_pointplot(self) -> None:
        self._product.add_product(
            key=self.get_pointplot,
            value=sns.pointplot(data=self.df).get_figure(),
        )

    def get_scatterplot(self) -> None:
        self._product.add_product(
            key=self.get_scatterplot,
            value=sns.scatterplot(data=self.df).get_figure(),
        )

    def get_swarmplot(self) -> None:
        self._product.add_product(
            key=self.get_swarmplot,
            value=sns.swarmplot(data=self.df).get_figure(),
        )

    def get_distributionplot(self) -> None:
        self._product.add_product(
            key=self.get_distributionplot,
            value=sns.displot(data=self.df).get_figure(),
        )

    def get_relationalplot(self) -> None:
        self._product.add_product(
            key=self.get_relationalplot,
            value=sns.relplot(data=self.df).get_figure(),
        )

    def get_categoryplot(self) -> None:
        self._product.add_product(
            key=self.get_categoryplot,
            value=sns.catplot(data=self.df).get_figure(),
        )

    def get_boxplot(self) -> None:
        self._product.add_product(
            key=self.get_boxplot,
            value=sns.boxplot(data=self.df).get_figure(),
        )

    def get_boxenplot(self) -> None:
        self._product.add_product(
            key=self.get_boxenplot,
            value=sns.boxenplot(data=self.df).get_figure(),
        )

    def get_stripplot(self) -> None:
        self._product.add_product(
            key=self.get_stripplot,
            value=sns.stripplot(data=self.df).get_figure(),
        )

    def get_hexagonplot(self) -> None:
        self._product.add_product(
            key=self.get_hexagonplot,
            value=sns.jointplot(data=self.df, kind="hex").get_figure(),
        )

    def get_histogramplot(self) -> None:
        self._product.add_product(
            key=self.get_histogramplot,
            value=sns.histplot(data=self.df).get_figure(),
        )

    def get_violinplot(self) -> None:
        self._product.add_product(
            key=self.get_violinplot,
            value=sns.violinplot(data=self.df).get_figure(),
        )

    def get_residualplot(self) -> None:
        self._product.add_product(
            key=self.get_residualplot,
            value=sns.residplot(data=self.df).get_figure(),
        )

    def get_regressionplot(self) -> None:
        self._product.add_product(
            key=self.get_regressionplot,
            value=sns.lmplot(data=self.df).get_figure(),
        )

    def get_density_mapplot(self) -> None:
        self._product.add_product(
            key=self.get_hexagonplot,
            value=sns.jointplot(data=self.df, kind="kde").get_figure(),
        )

    def get_kerneldensity_mapplot(self) -> None:
        self._product.add_product(
            key=self.get_kerneldensity_mapplot,
            value=sns.jointplot(data=self.df, kind="kde").get_figure(),
        )

    def get_cluster_mapplot(
        self, method: str = "pearson", min_periods: int = 1
    ) -> None:
        self._product.add_product(
            key=self.get_cluster_mapplot,
            value=sns.clustermap(
                data=self.df.corr(method=method, min_periods=min_periods)
            ).get_figure(),
        )

    def get_heatmapplot(self) -> None:
        self._product.add_product(
            key=self.get_cluster_mapplot,
            value=sns.heatmap(data=self.df).get_figure(),
        )

    def get_correlationpplot(
        self, method: str = "pearson", min_periods: int = 1
    ) -> None:
        self._product.add_product(
            key=self.get_correlationpplot,
            value=sns.relplot(
                data=self.df.corr(method=method, min_periods=min_periods)
            ).get_figure(),
        )

    def get_diagonal_correlationpplot(
        self, method: str = "pearson", min_periods: int = 1
    ) -> None:
        _corr = self.df.corr(method=method, min_periods=min_periods)
        _mask = np.triu(np.ones_like(_corr, dtype=bool))
        self._product.add_product(
            key=self.get_cluster_mapplot,
            value=sns.heatmap(data=_corr, mask=_mask).get_figure(),
        )

    def get_pairmapplot(self) -> None:
        self._product.add_product(
            key=self.get_pairmapplot,
            value=sns.pairplot(data=self.df).get_figure(),
        )

    def get_complex_pairmapplot(self) -> None:
        grid = sns.PairGrid(self.df, diag_sharey=False)
        grid.map_upper(sns.scatterplot, s=15)
        grid.map_lower(sns.kdeplot)
        grid.map_diag(sns.kdeplot, lw=2)
        self._product.add_product(
            key=self.get_pairmapplot,
            value=grid.get_figure(),
        )

    def get_regression_mapplot(self) -> None:
        self._product.add_product(
            key=self.get_regression_mapplot,
            value=sns.jointplot(data=self.df, kind="reg").get_figure(),
        )

    def get_pairmapplot(self) -> None:
        self._product.add_product(
            key=self.get_pairmapplot,
            value=sns.lmplot(data=self.df).get_figure(),
        )

    def get_nullity_matrix(self, **kwargs) -> None:
        """Generates the nullity matrix."""
        self._product.add_product(
            key=self.get_nullity_matrix,
            value=mss.matrix(self.df, **kwargs).get_figure(),
        )

    def get_nullity_bar(self, **kwargs) -> None:
        """Generates the nullity bar."""
        self._product.add_product(
            key=self.get_nullity_bar,
            value=mss.bar(self.df, **kwargs).get_figure(),
        )

    def get_nullity_heatmap(self, **kwargs) -> None:
        """Generates the nullity heatmap."""
        self._product.add_product(
            key=self.get_nullity_heatmap,
            value=mss.heatmap(self.df, cmap="seismic", **kwargs).get_figure(),
        )

    def get_nullity_dendrogram(self, **kwargs) -> None:
        """ Generates the nullity dendrogram."""
        self._product.add_product(
            key=self.get_nullity_dendrogram,
            value=mss.dendrogram(self.df, **kwargs).get_figure(),
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
