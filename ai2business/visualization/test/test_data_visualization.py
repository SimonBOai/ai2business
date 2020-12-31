"""Test-Environment for data_visualization."""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from ai2business.visualization import data_visualization as dav


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
        result[key].get_figure().savefig(Path(f"./test_visual_missing_data_{key}.png"))

    assert len(list(Path(".").glob("test_visual_missing_data_*.png"))) == i + 1


def test_list_product_parts():
    # Test return of prducts
    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(df_nan)
    data.builder = builder
    data.visual_missing_data()
    result = builder.data_figure.list_product_parts
    assert (
        result
        == "Product parts: get_nullity_matrix, get_nullity_bar, get_nullity_heatmap, get_nullity_dendrogram"
    )
