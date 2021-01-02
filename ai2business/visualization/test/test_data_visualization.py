"""Test-Environment for data_visualization."""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from ai2business.visualization import data_visualization as dav

from ai2business.macros import oneliner as one


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


def test_lineplot():
    # Test lineplot with new data set:
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(df_dict_years["get_interest_over_time"])
    data.builder = builder
    data.lineplot()
    builder.data_figure.save_all_figures()



df_dict_bigtech = one.TrendSearch.four_step_search(
    keyword_list=[
        "Apple",
        "Microsoft",
        "Google",
        "Huawei",
    ]
)

df_dict_corona = one.TrendSearch.four_step_search(
    keyword_list=[
        "Corona",
        "Vacination",
        "Vaccination",
        "Fear",
    ]
)