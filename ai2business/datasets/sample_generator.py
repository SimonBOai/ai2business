"""Generates sample list of KPIs"""
from ai2business.datasets.data import database


def stock_market(indices: str = "DOWJONES") -> dict:
    """Returns all company names and ISIN for a given stock market.

    Args:
        indices (str, optional): Name of the stock market. Defaults to "DOWJONES".

    Returns:
        dict: Collection of the indices of the current stock market.
    """
    try:
        return database.StockMarket.__dict__[indices.lower()]
    except KeyError as exc:
        print(f"ERROR: {exc} -> Indices is not listed in the databse!")
        return {}
