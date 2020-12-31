"""Test-Environment for data_visualization."""

from pathlib import Path

import seaborn as sns

from ai2business.visualization import data_visualization as dav


def test_visual_missing_data():
    data = dav.DataVisualization()
    iris = sns.load_dataset("iris").drop(columns="species")
    builder = dav.DesignerDataVisualization(iris)
    data.builder = builder
    data.visual_missing_data()
    result = builder.data_figure.return_product
    for i, key in enumerate(result.keys()):
        result[key].get_figure().savefig(Path(f"./nullity_{key}.png"))

    assert len(list(Path(".").glob("nullity*.png"))) == i + 1
