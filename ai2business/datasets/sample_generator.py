"""Generates sample list of KPIs"""
__all__ = ["stock_market"]
import json
from pathlib import Path

__cwd__ = Path("ai2business/kpi_collector/datasets/data/")


def load_json(database: str) -> dict:
    """Loading the database as JSON-files.

    Args:
        database (str): Name of the `JSON`-database.

    Returns:
        dict: Non-relational data as dictionary.
    """
    with open(Path(__cwd__).joinpath(database), "r") as file:
        return json.loads(file.read())


def stock_market(indices: str = "DOWJONES") -> dict:
    """Returns all company names and ISIN for a given stock market.

    Args:
        indices (str, optional): Name of the stock market. Defaults to "DOWJONES".

    Returns:
        dict: Collection of the indices of the current stock market.
    """
    try:
        return load_json("stock_market_index.json")[indices.upper()]
    except KeyError as exc:
        print(f"ERROR: {exc} -> Indices is not listed in the databse!")
