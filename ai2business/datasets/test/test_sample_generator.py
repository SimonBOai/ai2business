from ai2business.datasets import sample_generator
from ai2business.datasets.data import database

ref_DOW = database.StockMarket.__dict__["dowjones"]


def test_load_default() -> None:
    assert sample_generator.stock_market() == ref_DOW


def test_load_failed() -> None:
    assert sample_generator.stock_market("STOCK") == {}
