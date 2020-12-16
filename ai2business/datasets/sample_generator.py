"""Generates sample list of KPIs"""
from ai2business.datasets.data import database


class SampleGenerators:
    """Sample Generators allows to generate key word list.

    The module `sample_generator.py` contains functions, which allows generating a list of keywords
    with and without acronym:

    ```python
    # Get ticker values of the leading stock markert worldwide.
    stock_market(indices: str = "DOWJONES") -> dict
    ```
    """

    pass


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
        print(f"ERROR: {exc} -> Indices is not listed in the database!")
        return {}
