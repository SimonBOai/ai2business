"""Test-Environment for data_visualization."""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from ai2business.visualization import data_visualization as dav


def test_visual_missing_data():

    df = pd.DataFrame(
        np.random.randn(5, 3),
        index=["a", "c", "e", "f", "h"],
        columns=["one", "two", "three"],
    )
    df["four"] = "bar"
    df["five"] = df["one"] > 0
    df = df.reindex(["a", "b", "c", "d", "e", "f", "g", "h"])

    data = dav.DataVisualization()
    builder = dav.DesignerDataVisualization(df)
    data.builder = builder
    data.visual_missing_data()
    result = builder.data_figure.return_product
    for i, key in enumerate(result.keys()):
        result[key].get_figure().savefig(Path(f"./test_visual_missing_data_{key}.png"))

    assert len(list(Path(".").glob("test_visual_missing_data_*.png"))) == i + 1
