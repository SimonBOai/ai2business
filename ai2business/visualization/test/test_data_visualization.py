"""Test-Environment for data_visualization."""

from pathlib import Path

import numpy as np
import pandas as pd
from ai2business.macros import oneliner as one
from ai2business.visualization import data_visualization as dav

import matplotlib.pyplot as plt

df_nan = pd.DataFrame(
    np.random.randn(5, 3),
    index=["a", "c", "e", "f", "h"],
    columns=["one", "two", "three"],
)
df_nan["four"] = "bar"
df_nan["five"] = df_nan["one"] > 0
df_nan = df_nan.reindex(["a", "b", "c", "d", "e", "f", "g", "h"])


def test_visual_missing_data():
    # Test visualization of missing data
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
    # Test automatic saving of all figures
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(df_nan)
    data.builder = builder
    data.visual_missing_data()
    builder.data_figure.save_all_figures(folder="tmp")

    assert len(list(Path("tmp").glob("*.png"))) == 4


df_dict_years = one.TrendSearch.four_step_search(
    keyword_list=[
        "2017",
        "2018",
        "2019",
        "2020",
        "2021",
    ]
)


def test_lineplot_white():
    # Test lineplot with new data set:
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
    # Test lineplot with new data set:
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
    # Test lineplot with new data set:
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
    builder = dav.DesignerDataVisualization(df_dict_years["get_interest_over_time"])
    data.builder = builder
    data.lineplot()
    folder = f"{test_lineplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)

    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


df_dict_bigtech = one.TrendSearch.four_step_search(
    keyword_list=["Apple", "Google", "Smartphone", "Price"]
)


def test_pointplot():
    # Test pointplot with new data set:
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_bigtech["get_interest_over_time"],
        x_label="Apple",
        y_label="Google",
    )
    data.builder = builder
    data.pointplot()
    folder = f"{test_pointplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


def test_scatterplot():
    # Test scatterplot with new data set:
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(
        df_dict_bigtech["get_interest_over_time"],
        x_label="Apple",
        y_label="Google",
        hue="Smartphone",
        palette="ch:r=-.2,d=.3_r",
    )
    data.builder = builder
    data.scatterplot(size="Price")
    folder = f"{test_scatterplot.__name__}"
    builder.data_figure.save_all_figures(folder=folder)
    assert len(list(Path(f"{folder}").glob("*.png"))) == 1


df_dict_corona = one.TrendSearch.four_step_search(
    keyword_list=[
        "Corona",
        "Vaccination",
        "Hope",
        "Fear",
    ]
)
