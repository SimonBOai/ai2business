import json
from pathlib import Path

from ai2business.datasets.data import SampleGenerators, database

__cwd__ = Path("ai2business/datasets/data/")


def test_definition() -> None:
    assert SampleGenerators.__name__ == "SampleGenerators"


def test_dowjones_indices() -> None:
    with open(Path(__cwd__).joinpath("stock_market_indices.json"), "r") as file:
        ref_dict = json.loads(file.read())["dowjones"]

    assert database.StockMarket.__dict__["dowjones"] == ref_dict
