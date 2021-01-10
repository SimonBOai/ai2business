# Copyright 2020 AI2Business. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test-Environment for data_visualization."""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from ai2business.macros import oneliner as one
from ai2business.visualization import data_visualization as dav

df_nan = pd.DataFrame(
    np.random.randn(5, 3),
    index=["a", "c", "e", "f", "h"],
    columns=["one", "two", "three"],
)
df_nan["four"] = "bar"
df_nan["five"] = df_nan["one"] > 0
df_nan = df_nan.reindex(["a", "b", "c", "d", "e", "f", "g", "h"])

df_dict_fruits = one.TrendSearch.four_step_search(
    keyword_list=[
        "apple",
        "pineapple",
        "super market",
        "home delivery",
    ]
)

df_dict_years = one.TrendSearch.four_step_search(
    keyword_list=[
        "2017",
        "2018",
        "2019",
        "2020",
        "2021",
    ]
)


def test_visual_missing_data():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(df_nan)
    data.builder = builder
    data.visual_missing_data()
    result = builder.data_figure.return_product
    for i, key in enumerate(result.keys()):
        result[key].savefig(Path(f"./test_visual_missing_data_{key}.png"))

    assert len(list(Path(".").glob("test_visual_missing_data_*.png"))) == i + 1


def test_list_product_parts():
    # Test return of products
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(df_nan)
    data.builder = builder
    data.visual_missing_data()
    result = builder.data_figure.list_product_parts

    assert (
        result
        == "Product parts: get_nullity_matrix, get_nullity_bar, get_nullity_heatmap, get_nullity_dendrogram"
    )


def test_save_all_figures():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(df_nan)
    data.builder = builder
    data.visual_missing_data()
    builder.data_figure.save_all_figures(folder="tmp")

    assert len(list(Path("tmp").glob("*.png"))) == 4


def test_lineplot_white():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(df_dict_years["get_interest_over_time"])
    data.builder = builder
    data.lineplot()
    folder = "tmp_white"
    builder.data_figure.save_all_figures(folder=folder)

    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_lineplot_dark():
    # Test lineplot with new data set:
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_years["get_interest_over_time"],
        dark_mode=True,
    )
    data.builder = builder
    data.lineplot()
    folder = "tmp_dark"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_lineplot_whitegrid():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_years["get_interest_over_time"], grid=True
    )
    data.builder = builder
    data.lineplot()
    folder = "tmp_whitegrid"
    builder.data_figure.save_all_figures(folder=folder)

    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_lineplot_darkgrid():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_years["get_interest_over_time"], dark_mode=True, grid=True
    )
    data.builder = builder
    data.lineplot()
    folder = "tmp_darkgrid"
    builder.data_figure.save_all_figures(folder=folder)

    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_lineplot():
    # Test lineplot with new data set:
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(df_dict_fruits["get_interest_over_time"])
    data.builder = builder
    data.lineplot()
    folder = f"{test_lineplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)

    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_pointplot():
    # Test pointplot with new data set:
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"],
        x_label="apple",
        y_label="pineapple",
    )
    data.builder = builder
    data.pointplot()
    folder = f"{test_pointplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_scatterplot():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"],
        x_label="apple",
        y_label="pineapple",
        hue="super market",
        palette="ch:r=-.2,d=.3_r",
    )
    data.builder = builder
    data.scatterplot(size="home delivery")
    folder = f"{test_scatterplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_swarmplot():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"],
        palette="Set2",
    )
    data.builder = builder
    data.swarmplot(
        size=2,
        marker="D",
        edgecolor="gray",
        alpha=0.25,
    )
    folder = f"{test_swarmplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_distributionplot():
    # Test distributionplot with new data set:
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_related_queries"]["apple"]["top"],
        palette="dark",
        x_label="value",
    )
    data.builder = builder
    data.distributionplot(kind="ecdf")
    folder = f"{test_distributionplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_relationalplot():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"],
        palette="dark",
        x_label="apple",
        y_label="pineapple",
        hue="home delivery",
    )
    data.builder = builder
    data.relationalplot(col="super market")
    folder = f"{test_relationalplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_categoryplot():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"],
        palette="dark",
        x_label="apple",
        y_label="pineapple",
    )
    data.builder = builder
    data.categoryplot()
    folder = f"{test_categoryplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_boxplot():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"],
    )
    data.builder = builder
    data.boxplot()
    folder = f"{test_boxplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_boxenplot():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"],
        palette=sns.light_palette("purple"),
        dark_mode=True,
        grid=True,
    )
    data.builder = builder
    data.boxplot(multiboxen=True)
    folder = f"{test_boxenplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_stripplot():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"],
    )
    data.builder = builder
    data.stripplot()
    folder = f"{test_stripplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_hexagonplot():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"],
        x_label="apple",
        y_label="pineapple",
    )
    data.builder = builder
    data.hexagonplot(color="#4CB391")
    folder = f"{test_hexagonplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_histogramplot():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"], x_label="apple", hue="super market"
    )
    data.builder = builder
    data.histogramplot(multiple="stack", log_scale=True)
    folder = f"{test_histogramplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_violinplot():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"],
    )
    data.builder = builder
    data.violinplot()
    folder = f"{test_violinplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_residualplot():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"],
        x_label="apple",
        y_label="super market",
    )
    data.builder = builder
    data.residualplot(lowess=True, color="b")
    folder = f"{test_residualplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_regressionplot():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"],
        x_label="apple",
        y_label="pineapple",
    )
    data.builder = builder
    data.regressionplot()
    folder = f"{test_regressionplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_regressionmap():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"],
        x_label="apple",
        y_label="pineapple",
    )
    data.builder = builder
    data.regressionplot(map=True)
    folder = f"{test_regressionmap.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_densitymap():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"],
    )
    data.builder = builder
    data.densitymap()
    folder = f"{test_densitymap.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_kdemap():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"],
        x_label="apple",
        y_label="pineapple",
    )
    data.builder = builder
    data.densitymap(kde=True)
    folder = f"{test_kdemap.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_heatmap():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"].drop(columns="isPartial"),
        x_label="apple",
        y_label="pineapple",
    )
    data.builder = builder
    data.heatmap()
    folder = f"{test_heatmap.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_clustermap():
    df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

    # Select a subset of the networks
    used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
    used_columns = (
        df.columns.get_level_values("network").astype(int).isin(used_networks)
    )
    df = df.loc[:, used_columns]

    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(df)
    data.builder = builder
    data.clustermap()
    folder = f"{test_clustermap.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_correlation():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"],
        x_label="apple",
        y_label="pineapple",
    )
    data.builder = builder
    data.correlationmap()
    folder = f"{test_correlation.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_correlation_2():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"],
        x_label="apple",
        y_label="pineapple",
    )
    data.builder = builder
    data.correlationmap(diagonal=True)
    folder = f"{test_correlation_2.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_pairmap():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"].drop(columns="isPartial"),
        x_label="apple",
        y_label="pineapple",
    )
    data.builder = builder
    data.pairmap()
    folder = f"{test_pairmap.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_pairmap_complex():
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_fruits["get_interest_over_time"].drop(columns="isPartial"),
        x_label="apple",
        y_label="pineapple",
    )
    data.builder = builder
    data.pairmap(complex=True)
    folder = f"{test_pairmap_complex.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1
